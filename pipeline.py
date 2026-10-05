"""Reproducible modeling pipeline for synthetic HIV/ART experiments.

Task-specific preprocessing; identical, unaugmented records and unit weights;
fixed validation/test split; saved models, predictions, settings and checks.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.metadata
import json
import platform
import random
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (accuracy_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss, confusion_matrix, f1_score, log_loss, roc_auc_score)
from xgboost import XGBClassifier

BASE = Path(__file__).resolve().parent
NUM = ['AGEATVIS','HEIGHT','WEIGHT','BMI','TOTCHOL','HDL','TRIGLYC','NONHDL','MHLDLC',
       'ALBU','PROT','GLOBULIN','A_G_RATIO','TC_HDL_RATIO','LDL_HDL_RATIO','TG_HDL_RATIO',
       'HBA1C','GLUCOSE','CREAT','AST','ALT']
CAT = ['SEXBIRTH','FASTING','DIABETES']
ART_NUM = NUM + ['ART_DURATION_DAYS','CD4N','CD4P','CD8N','CD8P','HSRAT']
ART_CAT = CAT + ['CHOLRX','NIACIN','FIBRATES','ONART','ARTV','HAARTV2','ADH']
LIMITS = dict(zip(NUM,[(18,100),(1.2,2.2),(30,250),(10,80),(50,500),(5,200),(10,2000),
    (10,450),(5,400),(1,7),(2,12),(.5,8),(.1,5),(.5,20),(.1,15),(.1,50),
    (2,25),(30,700),(.1,20),(1,2000),(1,2000)]))
LIMITS.update(ART_DURATION_DAYS=(0,15000),CD4N=(0,5000),CD4P=(0,100),
              CD8N=(0,10000),CD8P=(0,100),HSRAT=(0,10))

def dump(path, obj):
    Path(path).write_text(json.dumps(obj,indent=2,default=lambda x:x.item() if hasattr(x,'item') else str(x)),encoding='utf-8')

def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def seed_all(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

class Preprocessor:
    def __init__(self, numerical, categorical, seed=42):
        self.requested_num=list(numerical); self.requested_cat=list(categorical); self.seed=seed

    def fit(self, frame):
        # Select task predictors before fitting preprocessing.
        frame=frame.loc[:,self.requested_num+self.requested_cat].copy()
        self.num=[c for c in self.requested_num if frame[c].isna().mean()<=.5]
        self.cat=[c for c in self.requested_cat if frame[c].isna().mean()<=.5]
        self.fences={}; self.outliers={}
        x=frame[self.num].apply(pd.to_numeric,errors='coerce').copy()
        for c in self.num:
            q1,q3=x[c].quantile([.25,.75]); self.fences[c]=(q1-3*(q3-q1),q3+3*(q3-q1))
            flo,fhi=self.fences[c];lo,hi=LIMITS[c]
            bad=((x[c]<flo)&(x[c]<lo))|((x[c]>fhi)&(x[c]>hi))
            self.outliers[c]=int(bad.sum());x.loc[bad,c]=np.nan
        self.log=[c for c in self.num if x[c].skew()>1 and x[c].min()>=0]
        x.loc[:,self.log]=np.log1p(x[self.log])
        self.imputer=IterativeImputer(random_state=self.seed,max_iter=15,initial_strategy='median',skip_complete=True)
        z=self.imputer.fit_transform(x)
        self.mean=z.mean(0);self.std=z.std(0);self.std[self.std==0]=1
        self.modes={};self.maps={}
        for c in self.cat:
            values=frame[c].dropna().astype(str); self.modes[c]=values.mode().iloc[0]
            self.maps[c]={v:i for i,v in enumerate(sorted(values.unique()))}
        return self

    def transform(self,frame):
        frame=frame.loc[:,self.num+self.cat]
        x=frame[self.num].apply(pd.to_numeric,errors='coerce').copy()
        for c in self.num:
            flo,fhi=self.fences[c];lo,hi=LIMITS[c]
            x.loc[((x[c]<flo)&(x[c]<lo))|((x[c]>fhi)&(x[c]>hi)),c]=np.nan
        x.loc[:,self.log]=np.log1p(x[self.log].clip(lower=0))
        xn=((self.imputer.transform(x)-self.mean)/self.std).astype(np.float32)
        xc=np.stack([frame[c].astype('string').fillna(self.modes[c]).astype(str)
            .map(self.maps[c]).fillna(len(self.maps[c])).to_numpy(np.int64) for c in self.cat],1)
        assert np.isfinite(xn).all()
        return xn,xc

    @property
    def cards(self): return [len(self.maps[c])+1 for c in self.cat]

    def report(self):
        return dict(numerical=self.num,categorical=self.cat,log1p=self.log,
            excluded_missing=[c for c in self.requested_num+self.requested_cat if c not in self.num+self.cat],
            outliers_set_missing=self.outliers,imputer=self.imputer.get_params(),
            categorical_missing='Training mode',categorical_unknown='Reserved unseen token',
            numerical_scaling='Training mean and population standard deviation after imputation',
            missingness_threshold=.5,training_rows=self.fit_n)

class Tokenizer(nn.Module):
    def __init__(self,n,cards):
        super().__init__();self.weight=nn.Parameter(torch.empty(n,64));nn.init.xavier_uniform_(self.weight)
        self.bias=nn.Parameter(torch.zeros(n,64));self.emb=nn.ModuleList([nn.Embedding(k,64) for k in cards])
        self.cls=nn.Parameter(torch.zeros(1,1,64))
    def forward(self,x,c):
        return torch.cat([self.cls.expand(len(x),-1,-1),x[:,:,None]*self.weight+self.bias,
                          torch.stack([e(c[:,i]) for i,e in enumerate(self.emb)],1)],1)

class Transformer(nn.Module):
    def __init__(self,specs):
        super().__init__();self.tokens=nn.ModuleDict({t:Tokenizer(n,k) for t,(n,k) in specs.items()})
        layer=nn.TransformerEncoderLayer(64,8,128,.2,activation='gelu',batch_first=True,norm_first=True)
        self.encoder=nn.TransformerEncoder(layer,3,norm=nn.LayerNorm(64),enable_nested_tensor=False)
        self.heads=nn.ModuleDict({t:nn.Linear(64,1) for t in specs})
    def forward(self,x,c,task):return self.heads[task](self.encoder(self.tokens[task](x,c))[:,0]).squeeze(1)

class MLP(nn.Module):
    def __init__(self,n):
        super().__init__();self.net=nn.Sequential(nn.Linear(n,128),nn.ReLU(),nn.Linear(128,64),nn.ReLU(),nn.Linear(64,1))
    def forward(self,x):return self.net(x).squeeze(1)

@torch.inference_mode()
def predict(model,x,c=None,task=None):
    model.eval();out=[]
    for i in range(0,len(x),256):
        z=torch.as_tensor(x[i:i+256],dtype=torch.float32)
        v=model(z) if c is None else model(z,torch.as_tensor(c[i:i+256],dtype=torch.long),task)
        out.append(torch.sigmoid(v).numpy())
    return np.concatenate(out)

def metrics(y,p):
    tn,fp,fn,tp=confusion_matrix(y,p>=.5,labels=[0,1]).ravel()
    return dict(n=len(y),prevalence=float(np.mean(y)),accuracy=accuracy_score(y,p>=.5),
        sensitivity=tp/(tp+fn),specificity=tn/(tn+fp),precision=tp/(tp+fp) if tp+fp else 0,
        f1=f1_score(y,p>=.5,zero_division=0),auroc=roc_auc_score(y,p),auprc=average_precision_score(y,p),
        balanced_accuracy=balanced_accuracy_score(y,p>=.5),brier=brier_score_loss(y,p),
        log_loss=log_loss(y,p,labels=[0,1]),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp))

def train_single(model,task,arr,y,out,name,args,is_mlp=False):
    if getattr(args,'resume',False) and (out/f'{name}_history.csv').exists():
        ck=torch.load(out/f'{name}.pt',weights_only=True);model.load_state_dict(ck['state_dict'])
        history=pd.read_csv(out/f'{name}_history.csv')
        print(f'Resumed completed {task} {name}',flush=True)
        return model,dict(selected_epoch=ck['epoch'],validation_loss=ck['validation_loss'],stopped_epoch=int(history.epoch.max()))
    opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-5)
    x,c=arr['train'];ds=TensorDataset(torch.tensor(x),torch.tensor(c),torch.tensor(y['train'],dtype=torch.float32))
    loader=DataLoader(ds,batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(args.seed))
    best=float('inf');stale=0;history=[];checkpoint=out/f'{name}.pt'
    for epoch in range(1,args.epochs+1):
        model.train();total=0
        for xn,xc,target in loader:
            opt.zero_grad();logits=model(xn) if is_mlp else model(xn,xc,task)
            loss=nn.functional.binary_cross_entropy_with_logits(logits,target);loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=loss.item()*len(target)
        xv,cv=arr['validation'];pv=predict(model,xv,None if is_mlp else cv,task)
        val=float(log_loss(y['validation'],pv,labels=[0,1]))
        history.append(dict(epoch=epoch,train_loss=total/len(ds),validation_loss=val))
        print(f'{task} {name} epoch {epoch}: validation loss {val:.6f}',flush=True)
        if val<best-1e-5:
            best=val;stale=0;torch.save(dict(state_dict=model.state_dict(),epoch=epoch,validation_loss=val),checkpoint)
        else:stale+=1
        if stale>=args.patience:break
    ck=torch.load(checkpoint,weights_only=True);model.load_state_dict(ck['state_dict'])
    pd.DataFrame(history).to_csv(out/f'{name}_history.csv',index=False)
    return model,dict(selected_epoch=ck['epoch'],validation_loss=ck['validation_loss'],stopped_epoch=epoch)

def train_multi(specs,data,parts,out,args):
    seed_all(args.seed);model=Transformer(specs);opt=torch.optim.Adam(model.parameters(),lr=.001,weight_decay=1e-5)
    if getattr(args,'resume',False) and (out/'multitask_history.csv').exists():
        ck=torch.load(out/'multitask.pt',weights_only=True);assert ck['specs']==specs
        model.load_state_dict(ck['state_dict']);history=pd.read_csv(out/'multitask_history.csv')
        print('Resumed completed multitask training',flush=True)
        return model,dict(selected_epoch=ck['epoch'],validation_loss=ck['validation_loss'],stopped_epoch=int(history.epoch.max()))
    # Both transforms applied independently to the same original rows; ART targets masked.
    xh,ch=data['HIV']['all']['train'];xa,ca=data['ART']['all']['train'];frame=parts['train']
    ds=TensorDataset(torch.tensor(xh),torch.tensor(ch),torch.tensor(xa),torch.tensor(ca),
        torch.tensor(frame.HIV_SCREEN_LABEL.to_numpy(np.float32)),
        torch.tensor(frame.ART_UNSUPPRESSED_LABEL.fillna(0).to_numpy(np.float32)),
        torch.tensor(frame.ART_MONITORING_ELIGIBLE.to_numpy(bool)))
    loader=DataLoader(ds,batch_size=64,shuffle=True,generator=torch.Generator().manual_seed(args.seed))
    best=float('inf');stale=0;history=[]
    for epoch in range(1,args.epochs+1):
        model.train();total=0
        for xh,ch,xa,ca,yh,ya,mask in loader:
            opt.zero_grad();lh=nn.functional.binary_cross_entropy_with_logits(model(xh,ch,'HIV'),yh)
            la=nn.functional.binary_cross_entropy_with_logits(model(xa[mask],ca[mask],'ART'),ya[mask]) if mask.any() else lh*0
            loss=lh+la;loss.backward();nn.utils.clip_grad_norm_(model.parameters(),1);opt.step();total+=loss.item()*len(yh)
        vals={t:float(log_loss(data[t]['y']['validation'],predict(model,*data[t]['arr']['validation'],t),labels=[0,1])) for t in specs}
        val=sum(vals.values());history.append(dict(epoch=epoch,train_minibatch_loss=total/len(ds),validation_loss=val,**{t+'_validation_loss':v for t,v in vals.items()}))
        print(f'Multitask epoch {epoch}: validation loss {val:.6f}',flush=True)
        if val<best-1e-5:
            best=val;stale=0;torch.save(dict(state_dict=model.state_dict(),specs=specs,epoch=epoch,validation_loss=val),out/'multitask.pt')
        else:stale+=1
        if stale>=args.patience:break
    ck=torch.load(out/'multitask.pt',weights_only=True);model.load_state_dict(ck['state_dict'])
    pd.DataFrame(history).to_csv(out/'multitask_history.csv',index=False)
    return model,dict(selected_epoch=ck['epoch'],validation_loss=ck['validation_loss'],stopped_epoch=epoch)

def prepare(args,out):
    df=pd.read_csv(args.data)
    assert len(df)==5844 and df.CCSID.is_unique and df.SYNTHETIC_FLAG.eq(1).all()
    assert set(df.DATA_SPLIT)=={'train','validation','test'}
    parts={s:df.loc[df.DATA_SPLIT.eq(s)].reset_index(drop=True) for s in ['train','validation','test']}
    assert [len(parts[s]) for s in parts]==[4090,877,877]
    assert df.HIV_SCREEN_LABEL.equals(df.HIVPOS)
    eligible=df.ART_MONITORING_ELIGIBLE.eq(1)
    assert np.array_equal(df.loc[eligible,'ART_UNSUPPRESSED_LABEL'],(df.loc[eligible,'VLOAD']>=1000).astype(int))
    df[['CCSID','DATA_SPLIT']].to_csv(out/'split_manifest.csv',index=False)
    data={};checks={};specs={}
    for task,num,cat in [('HIV',NUM,CAT),('ART',ART_NUM,ART_CAT)]:
        frames={s:f if task=='HIV' else f.loc[f.ART_MONITORING_ELIGIBLE.eq(1)].reset_index(drop=True) for s,f in parts.items()}
        prep=Preprocessor(num,cat,args.seed).fit(frames['train']);prep.fit_n=len(frames['train'])
        joblib.dump(prep,out/f'{task}_preprocessor.joblib');dump(out/f'{task}_preprocessing.json',prep.report())
        all_arr={s:prep.transform(f) for s,f in parts.items()};arr={s:prep.transform(f) for s,f in frames.items()}
        target='HIV_SCREEN_LABEL' if task=='HIV' else 'ART_UNSUPPRESSED_LABEL'
        y={s:f[target].to_numpy(int) for s,f in frames.items()}
        enc=OneHotEncoder(categories=[np.arange(k) for k in prep.cards],handle_unknown='ignore',sparse_output=False,dtype=np.float32).fit(arr['train'][1])
        design={s:np.c_[xn,enc.transform(xc)] for s,(xn,xc) in arr.items()}
        joblib.dump(enc,out/f'{task}_onehot.joblib')
        for s,f in frames.items():
            f[['CCSID',target]].to_csv(out/f'{task}_{s}_records.csv',index=False)
            np.savez_compressed(out/f'{task}_{s}_matrix.npz',numerical=arr[s][0],categorical=arr[s][1],y=y[s])
        specs[task]=(len(prep.num),prep.cards)
        data[task]=dict(prep=prep,frames=frames,arr=arr,all=all_arr,y=y,design=design)
    # Regression test: excluded predictors, labels and split fields cannot affect fitting or inference.
    blocked=[c for c in df.columns if c not in NUM+CAT]
    perturbed=parts['train'].copy()
    for c in blocked: perturbed[c]=0
    alternate=Preprocessor(NUM,CAT,args.seed).fit(perturbed)
    x1,c1=data['HIV']['arr']['validation'];changed=parts['validation'].copy()
    for c in blocked: changed[c]=0
    x2,c2=alternate.transform(changed)
    np.testing.assert_array_equal(x1,x2);np.testing.assert_array_equal(c1,c2)
    checks['screening_excluded_column_fit_and_inference_invariance']=True
    checks['screening_imputer_feature_names']=data['HIV']['prep'].imputer.feature_names_in_.tolist()
    checks['unique_participants_and_fixed_splits']=True;checks['target_definitions_verified']=True
    checks['data_sha256']=digest(args.data);checks['augmentation']='None';checks['class_and_sample_weights']='Unit for all models'
    checks['task_training_counts']={t:len(d['y']['train']) for t,d in data.items()}
    dump(out/'validation_checks.json',checks);return data,parts,specs

def run(args):
    torch.set_num_threads(args.threads);torch.use_deterministic_algorithms(True);seed_all(args.seed)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    environment={d.metadata['Name']:d.version for d in importlib.metadata.distributions()}
    dump(out/'environment.json',dict(python=platform.python_version(),platform=platform.platform(),packages=environment))
    (BASE/'requirements.lock.txt').write_text('\n'.join(f'{k}=={v}' for k,v in sorted(environment.items()))+'\n')
    config=dict(seed=args.seed,threads=args.threads,max_epochs=args.epochs,patience=args.patience,
        threshold=.5,augmentation='none',class_weights='none',sample_weights='unit',tuning='No search; fixed a priori settings',
        early_stopping='Official validation partition; minimum BCE improvement 1e-5; best checkpoint restored',
        optimizer=dict(name='Adam',lr=.001,weight_decay=1e-5,batch_size=64,gradient_clip_norm=1),
        transformer=dict(token_dimension=64,blocks=3,heads=8,ffn=128,dropout=.2,activation='gelu'),
        multitask=dict(task_loss_weights=dict(HIV=1,ART=1),mask='ART loss on eligible rows only',
            tokenizers='Separate task tokenizers and fitted preprocessors; shared encoder',
            validation_loss='Mean HIV BCE plus mean eligible ART BCE',
            training_loss='Sum of minibatch mean HIV BCE and minibatch mean eligible ART BCE'),
        data_sha256=digest(args.data),source_sha256=digest(__file__))
    if getattr(args,'resume',False):
        previous=json.loads((out/'protocol.json').read_text())
        for k in config:
            if k!='source_sha256':assert previous[k]==config[k],f'Resume configuration mismatch: {k}'
        dump(out/'resume_provenance.json',dict(previous_source_sha256=previous['source_sha256'],current_source_sha256=config['source_sha256'],
            reason='Windows sandbox denied joblib multiprocessing pipe; reuse completed neural fits without changing training settings'))
    dump(out/'protocol.json',config)
    data,parts,specs=prepare(args,out);rows=[];predictions=[];training={};fit_records=[]
    def record(task,name,pv,pt):
        d=data[task]
        for split,p in [('validation',pv),('test',pt)]:
            rows.append(dict(task=task,model=name,split=split,**metrics(d['y'][split],p)))
            predictions.extend(dict(task=task,model=name,split=split,CCSID=int(i),label=int(y),probability=float(p_),prediction=int(p_>=.5))
                for i,y,p_ in zip(d['frames'][split].CCSID,d['y'][split],p))
        fit_records.append(dict(task=task,model=name,n=len(d['y']['train']),augmentation='none',class_weight='none',
            training_records_sha256=digest(out/f'{task}_train_records.csv'),matrix_sha256=digest(out/f'{task}_train_matrix.npz')))
        pd.DataFrame(rows).to_csv(out/'metrics.csv',index=False);pd.DataFrame(predictions).to_csv(out/'predictions.csv',index=False)
        dump(out/'training_summary.json',training);pd.DataFrame(fit_records).to_csv(out/'comparison_training_manifest.csv',index=False)
    multi,training['multitask']=train_multi(specs,data,parts,out,args)
    for task,d in data.items():record(task,'Multitask FT-Transformer',predict(multi,*d['arr']['validation'],task),predict(multi,*d['arr']['test'],task))
    for task,d in data.items():
        seed_all(args.seed);single=Transformer({task:specs[task]})
        single,training[task+'_single_ft']=train_single(single,task,d['arr'],d['y'],out,task+'_single_ft',args)
        record(task,'Single-task FT-Transformer',predict(single,*d['arr']['validation'],task),predict(single,*d['arr']['test'],task))
        seed_all(args.seed);mlp=MLP(d['design']['train'].shape[1]);mlparr={s:(x,np.zeros((len(x),1),np.int64)) for s,x in d['design'].items()}
        mlp,training[task+'_mlp']=train_single(mlp,task,mlparr,d['y'],out,task+'_mlp',args,True)
        record(task,'MLP',predict(mlp,d['design']['validation']),predict(mlp,d['design']['test']))
        models={'Logistic regression':LogisticRegression(max_iter=2000,random_state=args.seed),
            'Random forest':RandomForestClassifier(n_estimators=500,min_samples_leaf=2,n_jobs=args.threads,random_state=args.seed),
            'XGBoost':XGBClassifier(n_estimators=500,max_depth=4,learning_rate=.03,subsample=.8,colsample_bytree=.8,
                scale_pos_weight=1,n_jobs=args.threads,random_state=args.seed,eval_metric='logloss')}
        for name,m in models.items():
            print(f'Fitting {task} {name}',flush=True);m.fit(d['design']['train'],d['y']['train'])
            slug=name.lower().replace(' ','_');joblib.dump(m,out/f'{task}_{slug}.joblib');dump(out/f'{task}_{slug}_parameters.json',m.get_params())
            record(task,name,m.predict_proba(d['design']['validation'])[:,1],m.predict_proba(d['design']['test'])[:,1])
    manifest=pd.DataFrame(fit_records)
    assert manifest.groupby('task').training_records_sha256.nunique().eq(1).all()
    assert manifest.groupby('task').matrix_sha256.nunique().eq(1).all()
    dump(out/'completion.json',dict(training_complete=True,models_per_task=6,shared_record_and_matrix_checks=True))
    print(pd.DataFrame(rows).query("split=='test'").to_string(index=False),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,default=BASE/'data/MWCCS_full_5844.csv')
    p.add_argument('--output',type=Path,default=BASE/'results');p.add_argument('--seed',type=int,default=42)
    p.add_argument('--epochs',type=int,default=100);p.add_argument('--patience',type=int,default=10);p.add_argument('--threads',type=int,default=4)
    run(p.parse_args())
