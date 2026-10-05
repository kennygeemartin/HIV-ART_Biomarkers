# HIV and ART Biomarker Modeling

A reproducible Python pipeline for HIV screening and ART monitoring using 5,844 synthetic MWCCS-style proxy records. The experiment compares a multitask FT-Transformer, single-task FT-Transformers, an MLP, logistic regression, random forest and XGBoost.

## Data

The dataset contains simulated records rather than actual MWCCS participant observations. The supplied CSV supports reproduction of model development; the upstream synthetic-data generator is unavailable. The data card describes fields and intended use.

Participant partitions are fixed: 4,090 training, 877 validation and 877 test records for HIV. ART modeling uses eligible subsets containing 2,296 training, 498 validation and 500 test records. HIV labels follow `HIVPOS`; the ART outcome is viral load at least 1,000 copies/mL among eligible records.

## Reproduce

Use Python 3.13 and install the recorded dependencies:

```sh
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.lock.txt
python run_experiment.py
python analyze_results.py
python verify_results.py
python summarize_results.py
```

To inspect the supplied models and outputs without retraining, run the last two commands. To regenerate figures, run `python analyze_results.py --stage plots`. Training defaults are seed 42, four CPU threads, 100 maximum epochs and early stopping patience 10. Numerical library builds and hardware can affect rerun results.

## Methods

Each task has a training-fitted preprocessor. HIV uses 21 numerical biomarker/demographic predictors and three categorical predictors: sex, fasting and diabetes. ART adds medication, treatment and immune-status predictors. Outcomes, identifiers and metadata are excluded from predictor fitting.

Preprocessing applies training-fitted outlier limits, log transforms for positively skewed nonnegative variables, iterative Bayesian-ridge imputation, numerical standardization and categorical encoding. Predictors with training missingness above 50% are excluded. The imputer uses at most 15 rounds; at least one fit reached this limit before satisfying its convergence criterion.

All models within a task use identical records and transformed inputs, with unit sample weights and unweighted classes. Transformers use categorical embeddings; other models use training-fitted one-hot encoding. The multitask model has separate task tokenizers, a shared encoder and two heads with equal loss weights. ART loss is evaluated on eligible records. Single-task transformers provide controls for auxiliary task learning.

The Transformer uses dimension 64, three blocks, eight attention heads, feed-forward dimension 128 and dropout 0.20. Neural models use Adam at learning rate 0.001, weight decay 0.00001 and batch size 64. Early stopping uses validation loss. All models use threshold 0.50 and fixed configurations. Complete settings are recorded in `results/protocol.json` and model parameter files.

## Results

| Task | Multitask AUROC | Multitask average precision | Test records |
| --- | --- | --- | --- |
| HIV | 0.7188 | 0.8134 | 877 |
| ART | 0.8875 | 0.5123 | 500 |

[RESULTS.md](RESULTS.md) contains all model comparisons, validation metrics, bootstrap intervals and SHAP summaries. The highest test AUROC is 0.7262 for HIV logistic regression and 0.9333 for the ART MLP.

SHAP uses permutation explanations, 20 training backgrounds and 60 test records per task, seed 42. Full arrays, record IDs, settings and additivity checks are supplied. Independent masking can disrupt correlated biomarkers; explanations describe model associations. Bootstrap intervals use 1,000 paired resamples of fixed test predictions and condition on fitted models.

These simulation results characterize a fixed dataset. Clinical effectiveness, predictor availability before diagnosis and performance on external patient cohorts require separate assessment.

## Repository contents

- `pipeline.py`, `run_experiment.py`: preprocessing, training and evaluation.
- `analyze_results.py`: plots, SHAP explanations and bootstrap intervals.
- `verify_results.py`: model reload, prediction, metric and explanation checks.
- `summarize_results.py`: generate the results report from saved outputs.
- `data/`: synthetic dataset and data card.
- `results/`: fitted models, predictions, matrices, figures, settings and verification records.
- `requirements.lock.txt`: recorded dependency versions.

The supplied dataset, checkpoints and numerical results are preserved. `results/source_manifest.json` records the training pipeline hash and published source hashes.
