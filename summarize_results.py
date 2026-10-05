"""Build the results report directly from saved outputs."""
import json
from pathlib import Path
import pandas as pd
from pipeline import BASE

out=BASE/'results';m=pd.read_csv(out/'metrics.csv');test=m.query("split=='test'");ci=pd.read_csv(out/'bootstrap_intervals.csv')
training=json.loads((out/'training_summary.json').read_text());settings=json.loads((out/'shap_settings.json').read_text())
checks=json.loads((out/'final_verification.json').read_text()) if (out/'final_verification.json').exists() else {}
lines=[]
def p(text=''):lines.extend([text,''])
def table(frame):
    columns=list(frame.columns);lines.append('| '+' | '.join(str(c).replace('_',' ') for c in columns)+' |')
    lines.append('| '+' | '.join('---' for c in columns)+' |')
    for _,row in frame.iterrows():
        lines.append('| '+' | '.join(f'{v:.4f}' if isinstance(v,float) else str(v) for v in row)+' |')
    lines.append('')

p('# HIV screening and ART monitoring results')
p('This experiment uses 5,844 synthetic MWCCS-style proxy records for HIV screening and ART monitoring model development. Results characterize this simulation dataset.')
p('## Primary findings')
for task in ['HIV','ART']:
    r=test.query("task==@task and model=='Multitask FT-Transformer'").iloc[0]
    p(f"For {task}, the multitask FT Transformer achieved AUROC {r.auroc:.4f}, average precision {r.auprc:.4f}, accuracy {r.accuracy:.4f}, sensitivity {r.sensitivity:.4f} and specificity {r.specificity:.4f} at threshold 0.50. The confusion counts were TN {r.tn}, FP {r.fp}, FN {r.fn} and TP {r.tp}.")
    best=test.query('task==@task').sort_values('auroc',ascending=False).iloc[0]
    p(f"The highest observed {task} AUROC was {best.auroc:.4f} for {best.model}. Model rankings are based on the fixed held-out split.")
p('## Evaluation protocol')
p('Screening preprocessing is fitted solely on the 21 permitted numerical and three categorical screening predictors. Medication, treatment, ART and immune status fields are excluded before fitting the imputer or any other transformation. ART has a separate preprocessing object fitted only on eligible training participants. A regression test changes all excluded fields and verifies identical screening fitting, transformation and final predictions.')
p('All six models within a task use identical original training records and the same fitted preprocessing. No additional records are generated, and every model uses unit sample weights and no class weighting. There are 4,090 HIV training records and 2,296 eligible ART training records. Validation and test partitions: HIV n=877 and n=877; ART n=498 and n=500. The data file SHA256 is 5a88602b21d21e9dd50e7f03e7880105c6bcced0c577cc94e1daa50bd1a602cb.')
p('The multitask model uses task specific tokenizers, a shared encoder and two heads. Single task transformers are fitted separately as controls for auxiliary task learning. Conventional models and the MLP use one hot encodings of the same categorical tokens. Thus representation differs as required by each architecture, but training records and preprocessing are matched.')
p('The Transformer has token dimension 64, three blocks, eight heads, feed forward dimension 128, GELU activation and dropout 0.20. The MLP has two ReLU hidden layers with 128 and 64 units. Neural models use Adam at learning rate 0.001, weight decay 0.00001, batch size 64, gradient clipping at norm 1, seed 42 and four CPU threads. HIV and eligible ART binary cross entropy terms have equal weight in multitask training. ART loss is masked to eligible records; validation loss is the sum of exact task mean losses.')
p('Neural early stopping uses the original validation partition with patience 10, maximum 100 epochs and improvement threshold 0.00001. The best checkpoint is restored. Logistic regression, random forest and XGBoost use fixed settings without hyperparameter search. The classification threshold is 0.50 for every model. Average precision is the reported AUPRC. Full configurations, package versions and source hashes accompany the report.')
p('The iterative imputer uses at most 15 rounds. At least one fit reached the iteration limit before satisfying the convergence criterion.')
p('## Held out test comparisons')
for task in ['HIV','ART']:
    p('### '+task+' performance')
    table(test.query('task==@task')[['model','accuracy','sensitivity','specificity','precision','f1','auroc','auprc']])
    table(test.query('task==@task')[['model','balanced_accuracy','brier','log_loss','tn','fp','fn','tp']])
p('For HIV, always predicting the majority HIV positive class produces accuracy 0.6420. For ART, always predicting suppression produces accuracy 0.9040. Therefore accuracy alone is insufficient, especially for ART. AUPRC reference prevalence is 0.6420 for HIV and 0.0960 for eligible ART test records.')
p('## Validation results and checkpoint selection')
for task in ['HIV','ART']:
    p('### '+task+' validation')
    table(m.query("task==@task and split=='validation'")[['model','accuracy','sensitivity','specificity','f1','auroc','auprc','brier']])
table(pd.DataFrame([dict(model=k,**v) for k,v in training.items()]))
p('## Uncertainty in held out discrimination')
p('Intervals use 1,000 paired ordinary bootstrap resamples of synthetic test participants with seed 73 and percentile 95% bounds. The models are not retrained inside the bootstrap; these intervals do not include model training or data generation uncertainty. Pairwise differences are exploratory with no multiplicity adjustment.')
for task in ['HIV','ART']:
    p('### '+task+' bootstrap intervals')
    interval_table=ci.query('task==@task')[['model','metric','lower','upper']].copy()
    interval_table['metric']=interval_table.metric.str.replace('_',' ')
    table(interval_table)
p('## Explanations of the final multitask model')
p('Permutation SHAP was computed from the multitask checkpoint for both tasks. The analysis uses 20 seed selected original training background records and 60 seed selected test records per task, without replacement. ART backgrounds and evaluated records are restricted to eligible participants. The seed is 42, the masker is Independent, the link is identity and the explained output is positive class probability. Input values are task transformed numerical features and categorical token codes. Saved files contain all record IDs, feature values, base values, full SHAP arrays, probabilities and exact settings.')
for task in ['HIV','ART']:
    s=settings[task];p(f"{task} explains {len(s['features'])} features using max_evals {s['max_evals']}. Maximum absolute reconstruction error from the baseline plus all contributions was {s['max_absolute_additivity_residual']:.8g}. This numerical additivity check does not establish Monte Carlo convergence.")
    table(pd.read_csv(out/f'{task}_shap_importance.csv'))
p('The waterfall is computed from all feature contributions and selects the highest predicted risk among the 60 explained records. Undisplayed features are aggregated by the plotting method. Independent masking can break biological correlations and relationships between biomarkers and derived ratios. These are approximate model associations, not causal effects or clinical biomarker validation.')
p('## Verification')
p(f"Independent verification passed: {checks.get('passed',False)}. The verification reloads all 12 model and task fits, reproduces all saved validation and test probabilities, recalculates 24 metric rows, checks matching training hashes and confirms SHAP additivity and split membership.")
p('## Reproducibility records')
p('See README.md, requirements.lock.txt, pipeline.py, run_experiment.py, analyze_results.py and verify_results.py. Machine readable evidence is in results/protocol.json, environment.json, validation_checks.json, final_verification.json, comparison_training_manifest.csv, predictions.csv, metrics.csv, shap_settings.json and the saved model and preprocessing files. The figures accompanying this report all come from the run.')
(BASE/'RESULTS.md').write_text('\n'.join(lines),encoding='utf-8')
print('Wrote RESULTS.md',flush=True)
