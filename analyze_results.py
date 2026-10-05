"""Generate evaluation plots, model explanations and uncertainty estimates."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parent/'.mplcache'))
Path(os.environ['MPLCONFIGDIR']).mkdir(parents=True,exist_ok=True)
import joblib
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,precision_recall_curve,roc_auc_score,average_precision_score
from sklearn.calibration import calibration_curve
import shap
from pipeline import BASE,Transformer,predict,dump,digest,seed_all

def curves(out):
    results=pd.read_csv(out/'metrics.csv');pred=pd.read_csv(out/'predictions.csv')
    for task in ['HIV','ART']:
        sub=pred.query("task==@task and split=='test'")
        fig,axs=plt.subplots(1,2,figsize=(11,4.5))
        for model,g in sub.groupby('model',sort=False):
            y=g.label.to_numpy();p=g.probability.to_numpy();fpr,tpr,_=roc_curve(y,p);pr,re,_=precision_recall_curve(y,p)
            axs[0].plot(fpr,tpr,label=f'{model} ({roc_auc_score(y,p):.3f})')
            axs[1].plot(re,pr,label=f'{model} ({average_precision_score(y,p):.3f})')
        axs[0].plot([0,1],[0,1],'k--',lw=1);axs[1].axhline(y.mean(),color='k',ls='--',lw=1)
        axs[0].set(xlabel='False positive rate',ylabel='Sensitivity',title=f'{task} ROC')
        axs[1].set(xlabel='Recall',ylabel='Precision',title=f'{task} precision recall')
        for ax in axs:ax.legend(fontsize=6.5)
        fig.tight_layout();fig.savefig(out/f'{task}_roc_pr.png',dpi=240);plt.close(fig)
        row=results.query("task==@task and split=='test' and model=='Multitask FT-Transformer'").iloc[0]
        cm=np.array([[row.tn,row.fp],[row.fn,row.tp]],int);fig,ax=plt.subplots(figsize=(5,4.5));ax.imshow(cm,cmap='Blues')
        for (i,j),v in np.ndenumerate(cm):ax.text(j,i,str(v),ha='center',va='center',color='white' if v>cm.max()/2 else 'black',fontsize=16)
        ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Negative','Positive'],yticklabels=['Negative','Positive'],xlabel='Predicted',ylabel='Observed',title=f'{task} multitask FT')
        fig.tight_layout();fig.savefig(out/f'{task}_confusion.png',dpi=240);plt.close(fig)
        fig,ax=plt.subplots(figsize=(6,5));calibration_rows=[]
        for model,g in sub.groupby('model',sort=False):
            f,m=calibration_curve(g.label,g.probability,n_bins=8,strategy='quantile');ax.plot(m,f,'o-',label=model,markersize=3)
            calibration_rows.extend(dict(model=model,mean_probability=float(a),fraction_positive=float(b)) for a,b in zip(m,f))
        ax.plot([0,1],[0,1],'k--');ax.set(xlabel='Mean predicted probability',ylabel='Observed positive proportion',title=f'{task} calibration on test records');ax.legend(fontsize=7)
        fig.tight_layout();fig.savefig(out/f'{task}_calibration.png',dpi=240);plt.close(fig)
        pd.DataFrame(calibration_rows).to_csv(out/f'{task}_calibration.csv',index=False)
    fig,ax=plt.subplots(figsize=(7,4.5));h=pd.read_csv(out/'multitask_history.csv')
    ax.plot(h.epoch,h.train_minibatch_loss,label='Training minibatch loss');ax.plot(h.epoch,h.validation_loss,label='Validation task mean loss')
    ck=torch.load(out/'multitask.pt',weights_only=True);ax.axvline(ck['epoch'],ls='--',color='k',label='Selected epoch')
    ax.set(xlabel='Epoch',ylabel='HIV BCE plus eligible ART BCE',title='Multitask learning curves');ax.legend()
    fig.tight_layout();fig.savefig(out/'multitask_learning.png',dpi=240);plt.close(fig)

def explain(out,args):
    ck=torch.load(out/'multitask.pt',map_location='cpu',weights_only=True);model=Transformer(ck['specs']);model.load_state_dict(ck['state_dict']);model.eval()
    reports={}
    for task in ['HIV','ART']:
        prep=joblib.load(out/f'{task}_preprocessor.joblib');features=prep.num+prep.cat;nnum=len(prep.num)
        train=np.load(out/f'{task}_train_matrix.npz');test=np.load(out/f'{task}_test_matrix.npz')
        tr=np.c_[train['numerical'],train['categorical']].astype(float);te=np.c_[test['numerical'],test['categorical']].astype(float)
        rng=np.random.default_rng(args.seed);bi=np.sort(rng.choice(len(tr),20,replace=False));ei=np.sort(rng.choice(len(te),args.samples,replace=False))
        background=tr[bi];explain_x=te[ei]
        def prediction(z):
            cats=np.rint(z[:,nnum:]).astype(np.int64)
            for j,k in enumerate(prep.cards):cats[:,j]=np.clip(cats[:,j],0,k-1)
            return predict(model,z[:,:nnum].astype(np.float32),cats,task)
        max_evals=args.permutations*(2*len(features)+1)
        explainer=shap.PermutationExplainer(prediction,shap.maskers.Independent(background,max_samples=20),feature_names=features,seed=args.seed)
        print(f'{task} SHAP: {len(ei)} records, {len(features)} features, max_evals {max_evals}',flush=True)
        sv=explainer(explain_x,max_evals=max_evals,batch_size=256,silent=True)
        probabilities=prediction(explain_x);residual=probabilities-(sv.base_values+sv.values.sum(1))
        assert np.max(np.abs(residual))<1e-5
        np.savez_compressed(out/f'{task}_shap_values.npz',values=sv.values,base_values=sv.base_values,data=explain_x,
            background=background,predictions=probabilities,feature_names=np.array(features),residual=residual)
        trids=pd.read_csv(out/f'{task}_train_records.csv');teids=pd.read_csv(out/f'{task}_test_records.csv')
        trids.iloc[bi].to_csv(out/f'{task}_shap_background_records.csv',index=False)
        evaluated=teids.iloc[ei].copy();evaluated['probability']=probabilities;evaluated['base_value']=sv.base_values;evaluated['additivity_residual']=residual
        evaluated.to_csv(out/f'{task}_shap_evaluated_records.csv',index=False)
        importance=pd.DataFrame(dict(feature=features,mean_abs_shap=np.abs(sv.values).mean(0))).sort_values('mean_abs_shap',ascending=False)
        importance.to_csv(out/f'{task}_shap_importance.csv',index=False)
        shap.plots.bar(sv,max_display=15,show=False);plt.title(f'{task} multitask model');plt.tight_layout();plt.savefig(out/f'{task}_shap_bar.png',dpi=240,bbox_inches='tight');plt.close()
        shap.plots.beeswarm(sv,max_display=15,show=False);plt.title(f'{task} probability scale SHAP');plt.tight_layout();plt.savefig(out/f'{task}_shap_beeswarm.png',dpi=240,bbox_inches='tight');plt.close()
        local=int(np.argmax(probabilities));shap.plots.waterfall(sv[local],max_display=15,show=False)
        plt.title(f'{task} highest predicted risk among explained records\nSynthetic ID {teids.iloc[ei[local]].CCSID}; p={probabilities[local]:.3f}')
        plt.tight_layout();plt.savefig(out/f'{task}_shap_waterfall.png',dpi=240,bbox_inches='tight');plt.close()
        reports[task]=dict(explainer='shap.PermutationExplainer',shap_version=shap.__version__,seed=args.seed,
            masker='Independent',background_count=20,background_population='Original task training records',
            evaluated_count=len(ei),evaluated_population='Untouched task test partition',selection='Seeded uniform sampling without replacement; original order retained',
            output_scale='Positive class probability with identity link',input_space='Task-specific transformed numerical and ordinal categorical tokens',
            features=features,max_evals=max_evals,permutation_budget_multiplier=args.permutations,batch_size=256,
            model_sha256=digest(out/'multitask.pt'),preprocessor_sha256=digest(out/f'{task}_preprocessor.joblib'),
            max_absolute_additivity_residual=float(np.max(np.abs(residual))),
            local_waterfall_selection='Highest predicted positive probability among evaluated records',
            limitations='Approximate marginal explanations. Independent masking breaks biomarker dependencies and derived-ratio consistency; not causal. No SHAP convergence claim.')
        dump(out/'shap_settings.json',reports);print(f'{task} SHAP complete; residual {np.max(np.abs(residual)):.3g}',flush=True)

def saved_explanation_plots(out):
    """Render explanation figures from the saved SHAP arrays."""
    for task in ['HIV','ART']:
        arrays=np.load(out/f'{task}_shap_values.npz')
        sv=shap.Explanation(values=arrays['values'],base_values=arrays['base_values'],
            data=arrays['data'],feature_names=arrays['feature_names'].tolist())
        shap.plots.bar(sv,max_display=15,show=False)
        plt.title(f'{task} multitask model');plt.tight_layout()
        plt.savefig(out/f'{task}_shap_bar.png',dpi=240,bbox_inches='tight');plt.close()

def uncertainty(out):
    pred=pd.read_csv(out/'predictions.csv');rows=[]
    for task in ['HIV','ART']:
        sub=pred.query("task==@task and split=='test'");models=list(sub.model.unique())
        pivot=sub.pivot(index='CCSID',columns='model',values='probability');truth=sub.drop_duplicates('CCSID').set_index('CCSID').loc[pivot.index,'label'].to_numpy()
        values={m:[] for m in models};diff={m:[] for m in models if m!='Single-task FT-Transformer'};rng=np.random.default_rng(73)
        for _ in range(1000):
            ix=rng.integers(0,len(truth),len(truth));y=truth[ix]
            if len(np.unique(y))<2:continue
            aucs={}
            for m in models:
                p=pivot[m].to_numpy()[ix];auc=roc_auc_score(y,p);ap=average_precision_score(y,p)
                values[m].append((auc,ap));aucs[m]=auc
            for m in diff:diff[m].append(aucs[m]-aucs['Single-task FT-Transformer'])
        for m,vals in values.items():
            for j,metric in enumerate(['auroc','auprc']):
                low,high=np.quantile(np.array(vals)[:,j],[.025,.975]);rows.append(dict(task=task,model=m,metric=metric,lower=low,upper=high,bootstrap_replicates=len(vals)))
        for m,vals in diff.items():
            low,high=np.quantile(vals,[.025,.975]);rows.append(dict(task=task,model=m,metric='auroc_difference_vs_single_ft',lower=low,upper=high,bootstrap_replicates=len(vals)))
    pd.DataFrame(rows).to_csv(out/'bootstrap_intervals.csv',index=False)
    dump(out/'bootstrap_settings.json',dict(seed=73,replicates=1000,unit='Synthetic participant',method='Paired ordinary percentile bootstrap of held-out predictions',
        confidence_level=.95,training_repeated=False,scope='Conditional on fitted models and fixed dataset; not clinical validation; comparisons exploratory, no multiplicity adjustment'))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,default=BASE/'results');ap.add_argument('--seed',type=int,default=42)
    ap.add_argument('--samples',type=int,default=60);ap.add_argument('--permutations',type=int,default=5);ap.add_argument('--threads',type=int,default=4)
    ap.add_argument('--stage',choices=['all','plots','shap','shap-plots','uncertainty'],default='all');a=ap.parse_args()
    torch.set_num_threads(a.threads);seed_all(a.seed)
    if a.stage in ['all','plots']:curves(a.output)
    if a.stage in ['all','shap']:explain(a.output,a)
    if a.stage=='shap-plots':saved_explanation_plots(a.output)
    if a.stage in ['all','uncertainty']:uncertainty(a.output)
