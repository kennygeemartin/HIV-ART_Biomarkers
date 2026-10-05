"""Verify saved outputs independently and reload every fitted model."""
import json
import json
import joblib
import numpy as np
import pandas as pd
import torch
from pipeline import BASE,Transformer,MLP,NUM,CAT,metrics,predict,dump,digest

out=BASE/'results';torch.set_num_threads(4)
pred=pd.read_csv(out/'predictions.csv');table=pd.read_csv(out/'metrics.csv')
data=pd.read_csv(BASE/'data/MWCCS_full_5844.csv');manifest=pd.read_csv(out/'comparison_training_manifest.csv')
assert len(table)==24 and len(manifest)==12
assert manifest.groupby('task').training_records_sha256.nunique().eq(1).all()
assert manifest.groupby('task').matrix_sha256.nunique().eq(1).all()
errors=[];reload_errors=[];reload_metric_differences=[]
ck=torch.load(out/'multitask.pt',weights_only=True);multi=Transformer(ck['specs']);multi.load_state_dict(ck['state_dict'])
for task in ['HIV','ART']:
    prep=joblib.load(out/f'{task}_preprocessor.joblib');enc=joblib.load(out/f'{task}_onehot.joblib')
    ft=Transformer({task:(len(prep.num),prep.cards)});ft.load_state_dict(torch.load(out/f'{task}_single_ft.pt',weights_only=True)['state_dict'])
    for split in ['validation','test']:
        frame=data.loc[data.DATA_SPLIT.eq(split)].copy()
        if task=='ART':frame=frame.loc[frame.ART_MONITORING_ELIGIBLE.eq(1)]
        xn,xc=prep.transform(frame);design=np.c_[xn,enc.transform(xc)]
        mlp=MLP(design.shape[1]);mlp.load_state_dict(torch.load(out/f'{task}_mlp.pt',weights_only=True)['state_dict'])
        for name in manifest.loc[manifest.task.eq(task),'model']:
            if name=='Multitask FT-Transformer':p=predict(multi,xn,xc,task)
            elif name=='Single-task FT-Transformer':p=predict(ft,xn,xc,task)
            elif name=='MLP':p=predict(mlp,design)
            else:p=joblib.load(out/f"{task}_{name.lower().replace(' ','_')}.joblib").predict_proba(design)[:,1]
            stored=pred.query('task==@task and split==@split and model==@name').set_index('CCSID').loc[frame.CCSID]
            np.testing.assert_allclose(stored.probability,p,rtol=0,atol=1e-7)
            computed=metrics(stored.label,stored.probability.to_numpy());row=table.query('task==@task and split==@split and model==@name').iloc[0]
            for key,val in computed.items():np.testing.assert_allclose(row[key],val,rtol=0,atol=1e-7)
            # Parallel tree summation can break floating-point ties in rank metrics.
            reloaded_metrics=metrics(stored.label,p)
            reload_metric_differences.append(dict(task=task,split=split,model=name,
                differences={key:float(reloaded_metrics[key]-val) for key,val in computed.items()
                    if abs(reloaded_metrics[key]-val)>1e-7}))
            reload_errors.append(float(np.max(np.abs(stored.probability.to_numpy()-p))))
        if task=='HIV':
            modified=frame.copy()
            for c in data.columns:
                if c not in NUM+CAT:modified[c]=0
            xn2,xc2=prep.transform(modified)
            np.testing.assert_array_equal(xn,xn2);np.testing.assert_array_equal(xc,xc2)
            np.testing.assert_array_equal(predict(multi,xn,xc,'HIV'),predict(multi,xn2,xc2,'HIV'))
    sv=np.load(out/f'{task}_shap_values.npz')
    settings=json.loads((out/'shap_settings.json').read_text())[task]
    assert settings['model_sha256']==digest(out/'multitask.pt')
    assert settings['preprocessor_sha256']==digest(out/f'{task}_preprocessor.joblib')
    err=float(np.max(np.abs(sv['base_values']+sv['values'].sum(1)-sv['predictions'])))
    assert err<1e-5;errors.append(err)
    background=pd.read_csv(out/f'{task}_shap_background_records.csv');evaluated=pd.read_csv(out/f'{task}_shap_evaluated_records.csv')
    assert set(background.CCSID)<=set(data.loc[data.DATA_SPLIT.eq('train'),'CCSID'])
    assert set(evaluated.CCSID)<=set(data.loc[data.DATA_SPLIT.eq('test'),'CCSID'])
    assert not set(background.CCSID)&set(evaluated.CCSID)
    if task=='ART':assert set(background.CCSID)|set(evaluated.CCSID)<=set(data.loc[data.ART_MONITORING_ELIGIBLE.eq(1),'CCSID'])
dump(out/'reload_metric_differences.json',reload_metric_differences)
dump(out/'final_verification.json',dict(passed=True,metrics_rows_verified=24,model_task_fits_verified=12,
    every_model_reloaded=True,max_prediction_reload_error=max(reload_errors),max_shap_additivity_error=max(errors),
    matching_training_participant_and_matrix_hashes=True,excluded_screening_fields_prediction_invariance=True,
    shap_background_train_only=True,shap_evaluated_test_only=True,data_sha256=digest(BASE/'data/MWCCS_full_5844.csv')))
print('PASS: all 12 model/task fits reloaded; 24 metric rows and all saved predictions independently verified.',flush=True)
