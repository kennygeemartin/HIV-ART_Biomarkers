# HIV screening and ART monitoring results

This experiment uses 5,844 synthetic MWCCS-style proxy records for HIV screening and ART monitoring model development. Results characterize this simulation dataset.

## Primary findings

For HIV, the multitask FT Transformer achieved AUROC 0.7188, average precision 0.8134, accuracy 0.6921, sensitivity 0.8792 and specificity 0.3567 at threshold 0.50. The confusion counts were TN 112, FP 202, FN 68 and TP 495.

The highest observed HIV AUROC was 0.7262 for Logistic regression. Model rankings are based on the fixed held-out split.

For ART, the multitask FT Transformer achieved AUROC 0.8875, average precision 0.5123, accuracy 0.9140, sensitivity 0.5000 and specificity 0.9580 at threshold 0.50. The confusion counts were TN 433, FP 19, FN 24 and TP 24.

The highest observed ART AUROC was 0.9333 for MLP. Model rankings are based on the fixed held-out split.

## Evaluation protocol

Screening preprocessing is fitted solely on the 21 permitted numerical and three categorical screening predictors. Medication, treatment, ART and immune status fields are excluded before fitting the imputer or any other transformation. ART has a separate preprocessing object fitted only on eligible training participants. A regression test changes all excluded fields and verifies identical screening fitting, transformation and final predictions.

All six models within a task use identical original training records and the same fitted preprocessing. No additional records are generated, and every model uses unit sample weights and no class weighting. There are 4,090 HIV training records and 2,296 eligible ART training records. Validation and test partitions: HIV n=877 and n=877; ART n=498 and n=500. The data file SHA256 is 5a88602b21d21e9dd50e7f03e7880105c6bcced0c577cc94e1daa50bd1a602cb.

The multitask model uses task specific tokenizers, a shared encoder and two heads. Single task transformers are fitted separately as controls for auxiliary task learning. Conventional models and the MLP use one hot encodings of the same categorical tokens. Thus representation differs as required by each architecture, but training records and preprocessing are matched.

The Transformer has token dimension 64, three blocks, eight heads, feed forward dimension 128, GELU activation and dropout 0.20. The MLP has two ReLU hidden layers with 128 and 64 units. Neural models use Adam at learning rate 0.001, weight decay 0.00001, batch size 64, gradient clipping at norm 1, seed 42 and four CPU threads. HIV and eligible ART binary cross entropy terms have equal weight in multitask training. ART loss is masked to eligible records; validation loss is the sum of exact task mean losses.

Neural early stopping uses the original validation partition with patience 10, maximum 100 epochs and improvement threshold 0.00001. The best checkpoint is restored. Logistic regression, random forest and XGBoost use fixed settings without hyperparameter search. The classification threshold is 0.50 for every model. Average precision is the reported AUPRC. Full configurations, package versions and source hashes accompany the report.

The iterative imputer uses at most 15 rounds. At least one fit reached the iteration limit before satisfying the convergence criterion.

## Held out test comparisons

### HIV performance

| model | accuracy | sensitivity | specificity | precision | f1 | auroc | auprc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Multitask FT-Transformer | 0.6921 | 0.8792 | 0.3567 | 0.7102 | 0.7857 | 0.7188 | 0.8134 |
| Single-task FT-Transformer | 0.6830 | 0.8863 | 0.3185 | 0.6999 | 0.7821 | 0.7095 | 0.8155 |
| MLP | 0.6830 | 0.8455 | 0.3917 | 0.7136 | 0.7740 | 0.7217 | 0.8198 |
| Logistic regression | 0.6990 | 0.8668 | 0.3981 | 0.7208 | 0.7871 | 0.7262 | 0.8232 |
| Random forest | 0.6921 | 0.8757 | 0.3631 | 0.7114 | 0.7850 | 0.7246 | 0.8140 |
| XGBoost | 0.7035 | 0.8650 | 0.4140 | 0.7258 | 0.7893 | 0.7241 | 0.8154 |

| model | balanced accuracy | brier | log loss | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Multitask FT-Transformer | 0.6180 | 0.1986 | 0.5808 | 112 | 202 | 68 | 495 |
| Single-task FT-Transformer | 0.6024 | 0.2015 | 0.5865 | 100 | 214 | 64 | 499 |
| MLP | 0.6186 | 0.1973 | 0.5755 | 123 | 191 | 87 | 476 |
| Logistic regression | 0.6324 | 0.1960 | 0.5727 | 125 | 189 | 75 | 488 |
| Random forest | 0.6194 | 0.1971 | 0.5774 | 114 | 200 | 70 | 493 |
| XGBoost | 0.6395 | 0.1979 | 0.5801 | 130 | 184 | 76 | 487 |

### ART performance

| model | accuracy | sensitivity | specificity | precision | f1 | auroc | auprc |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Multitask FT-Transformer | 0.9140 | 0.5000 | 0.9580 | 0.5581 | 0.5275 | 0.8875 | 0.5123 |
| Single-task FT-Transformer | 0.9200 | 0.2917 | 0.9867 | 0.7000 | 0.4118 | 0.9180 | 0.5485 |
| MLP | 0.9320 | 0.4583 | 0.9823 | 0.7333 | 0.5641 | 0.9333 | 0.6116 |
| Logistic regression | 0.9340 | 0.5000 | 0.9801 | 0.7273 | 0.5926 | 0.9278 | 0.6353 |
| Random forest | 0.9420 | 0.4375 | 0.9956 | 0.9130 | 0.5915 | 0.9256 | 0.6497 |
| XGBoost | 0.9340 | 0.4583 | 0.9845 | 0.7586 | 0.5714 | 0.9260 | 0.6490 |

| model | balanced accuracy | brier | log loss | tn | fp | fn | tp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Multitask FT-Transformer | 0.7290 | 0.0662 | 0.2231 | 433 | 19 | 24 | 24 |
| Single-task FT-Transformer | 0.6392 | 0.0601 | 0.1965 | 446 | 6 | 34 | 14 |
| MLP | 0.7203 | 0.0532 | 0.1757 | 444 | 8 | 26 | 22 |
| Logistic regression | 0.7400 | 0.0530 | 0.1783 | 443 | 9 | 24 | 24 |
| Random forest | 0.7165 | 0.0530 | 0.1854 | 450 | 2 | 27 | 21 |
| XGBoost | 0.7214 | 0.0554 | 0.1876 | 445 | 7 | 26 | 22 |

For HIV, always predicting the majority HIV positive class produces accuracy 0.6420. For ART, always predicting suppression produces accuracy 0.9040. Therefore accuracy alone is insufficient, especially for ART. AUPRC reference prevalence is 0.6420 for HIV and 0.0960 for eligible ART test records.

## Validation results and checkpoint selection

### HIV validation

| model | accuracy | sensitivity | specificity | f1 | auroc | auprc | brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Multitask FT-Transformer | 0.7001 | 0.8668 | 0.4013 | 0.7877 | 0.7038 | 0.7863 | 0.2005 |
| Single-task FT-Transformer | 0.6864 | 0.8810 | 0.3376 | 0.7830 | 0.6924 | 0.7774 | 0.2047 |
| MLP | 0.7149 | 0.8721 | 0.4331 | 0.7971 | 0.7139 | 0.7899 | 0.1981 |
| Logistic regression | 0.7058 | 0.8668 | 0.4172 | 0.7909 | 0.7123 | 0.7898 | 0.1992 |
| Random forest | 0.7047 | 0.8774 | 0.3949 | 0.7923 | 0.7016 | 0.7901 | 0.2012 |
| XGBoost | 0.6990 | 0.8455 | 0.4363 | 0.7829 | 0.6918 | 0.7751 | 0.2061 |

### ART validation

| model | accuracy | sensitivity | specificity | f1 | auroc | auprc | brier |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Multitask FT-Transformer | 0.9036 | 0.3889 | 0.9662 | 0.4667 | 0.9186 | 0.5658 | 0.0688 |
| Single-task FT-Transformer | 0.9076 | 0.3148 | 0.9797 | 0.4250 | 0.9111 | 0.5674 | 0.0663 |
| MLP | 0.9197 | 0.4259 | 0.9797 | 0.5349 | 0.9181 | 0.6399 | 0.0620 |
| Logistic regression | 0.9137 | 0.4444 | 0.9707 | 0.5275 | 0.9176 | 0.6382 | 0.0633 |
| Random forest | 0.9116 | 0.2778 | 0.9887 | 0.4054 | 0.9189 | 0.6118 | 0.0642 |
| XGBoost | 0.9116 | 0.3889 | 0.9752 | 0.4884 | 0.9228 | 0.6116 | 0.0664 |

| model | selected epoch | validation loss | stopped epoch |
| --- | --- | --- | --- |
| multitask | 4 | 0.8142 | 14 |
| HIV_single_ft | 4 | 0.5973 | 14 |
| HIV_mlp | 5 | 0.5853 | 15 |
| ART_single_ft | 3 | 0.2195 | 13 |
| ART_mlp | 6 | 0.2054 | 16 |

## Uncertainty in held out discrimination

Intervals use 1,000 paired ordinary bootstrap resamples of synthetic test participants with seed 73 and percentile 95% bounds. The models are not retrained inside the bootstrap; these intervals do not include model training or data generation uncertainty. Pairwise differences are exploratory with no multiplicity adjustment.

### HIV bootstrap intervals

| model | metric | lower | upper |
| --- | --- | --- | --- |
| Multitask FT-Transformer | auroc | 0.6836 | 0.7530 |
| Multitask FT-Transformer | auprc | 0.7783 | 0.8470 |
| Single-task FT-Transformer | auroc | 0.6738 | 0.7434 |
| Single-task FT-Transformer | auprc | 0.7830 | 0.8469 |
| MLP | auroc | 0.6864 | 0.7553 |
| MLP | auprc | 0.7851 | 0.8515 |
| Logistic regression | auroc | 0.6914 | 0.7599 |
| Logistic regression | auprc | 0.7892 | 0.8548 |
| Random forest | auroc | 0.6908 | 0.7565 |
| Random forest | auprc | 0.7750 | 0.8479 |
| XGBoost | auroc | 0.6907 | 0.7567 |
| XGBoost | auprc | 0.7817 | 0.8470 |
| Multitask FT-Transformer | auroc difference vs single ft | -0.0036 | 0.0201 |
| MLP | auroc difference vs single ft | -0.0071 | 0.0321 |
| Logistic regression | auroc difference vs single ft | -0.0007 | 0.0339 |
| Random forest | auroc difference vs single ft | -0.0069 | 0.0383 |
| XGBoost | auroc difference vs single ft | -0.0085 | 0.0396 |

### ART bootstrap intervals

| model | metric | lower | upper |
| --- | --- | --- | --- |
| Multitask FT-Transformer | auroc | 0.8203 | 0.9376 |
| Multitask FT-Transformer | auprc | 0.3772 | 0.6547 |
| Single-task FT-Transformer | auroc | 0.8865 | 0.9451 |
| Single-task FT-Transformer | auprc | 0.4133 | 0.6811 |
| MLP | auroc | 0.9006 | 0.9578 |
| MLP | auprc | 0.4697 | 0.7659 |
| Logistic regression | auroc | 0.8904 | 0.9570 |
| Logistic regression | auprc | 0.4932 | 0.7735 |
| Random forest | auroc | 0.8833 | 0.9583 |
| Random forest | auprc | 0.5069 | 0.7885 |
| XGBoost | auroc | 0.8881 | 0.9544 |
| XGBoost | auprc | 0.5138 | 0.7636 |
| Multitask FT-Transformer | auroc difference vs single ft | -0.0842 | 0.0053 |
| MLP | auroc difference vs single ft | -0.0001 | 0.0320 |
| Logistic regression | auroc difference vs single ft | -0.0095 | 0.0284 |
| Random forest | auroc difference vs single ft | -0.0213 | 0.0330 |
| XGBoost | auroc difference vs single ft | -0.0129 | 0.0276 |

## Explanations of the final multitask model

Permutation SHAP was computed from the multitask checkpoint for both tasks. The analysis uses 20 seed selected original training background records and 60 seed selected test records per task, without replacement. ART backgrounds and evaluated records are restricted to eligible participants. The seed is 42, the masker is Independent, the link is identity and the explained output is positive class probability. Input values are task transformed numerical features and categorical token codes. Saved files contain all record IDs, feature values, base values, full SHAP arrays, probabilities and exact settings.

HIV explains 24 features using max_evals 245. Maximum absolute reconstruction error from the baseline plus all contributions was 6.1094761e-08. This numerical additivity check does not establish Monte Carlo convergence.

| feature | mean abs shap |
| --- | --- |
| TRIGLYC | 0.0751 |
| GLOBULIN | 0.0488 |
| A_G_RATIO | 0.0402 |
| ALBU | 0.0293 |
| TG_HDL_RATIO | 0.0282 |
| TOTCHOL | 0.0224 |
| MHLDLC | 0.0170 |
| PROT | 0.0143 |
| HDL | 0.0140 |
| NONHDL | 0.0132 |
| BMI | 0.0125 |
| LDL_HDL_RATIO | 0.0110 |
| HBA1C | 0.0087 |
| GLUCOSE | 0.0086 |
| AGEATVIS | 0.0085 |
| ALT | 0.0061 |
| TC_HDL_RATIO | 0.0058 |
| WEIGHT | 0.0058 |
| AST | 0.0055 |
| CREAT | 0.0048 |
| DIABETES | 0.0045 |
| HEIGHT | 0.0038 |
| SEXBIRTH | 0.0033 |
| FASTING | 0.0015 |

ART explains 37 features using max_evals 375. Maximum absolute reconstruction error from the baseline plus all contributions was 2.0582229e-09. This numerical additivity check does not establish Monte Carlo convergence.

| feature | mean abs shap |
| --- | --- |
| CD4N | 0.0563 |
| HSRAT | 0.0444 |
| ADH | 0.0404 |
| A_G_RATIO | 0.0317 |
| CD4P | 0.0244 |
| ALBU | 0.0213 |
| GLOBULIN | 0.0194 |
| TRIGLYC | 0.0141 |
| TG_HDL_RATIO | 0.0113 |
| AST | 0.0099 |
| CD8N | 0.0095 |
| HDL | 0.0084 |
| CD8P | 0.0082 |
| ART_DURATION_DAYS | 0.0081 |
| LDL_HDL_RATIO | 0.0075 |
| ARTV | 0.0071 |
| CREAT | 0.0070 |
| PROT | 0.0068 |
| MHLDLC | 0.0067 |
| WEIGHT | 0.0062 |
| HAARTV2 | 0.0060 |
| TOTCHOL | 0.0054 |
| HEIGHT | 0.0052 |
| NONHDL | 0.0049 |
| BMI | 0.0047 |
| GLUCOSE | 0.0045 |
| HBA1C | 0.0042 |
| TC_HDL_RATIO | 0.0042 |
| FASTING | 0.0040 |
| AGEATVIS | 0.0032 |
| ALT | 0.0031 |
| CHOLRX | 0.0020 |
| SEXBIRTH | 0.0010 |
| DIABETES | 0.0004 |
| NIACIN | 0.0003 |
| FIBRATES | 0.0000 |
| ONART | 0.0000 |

The waterfall is computed from all feature contributions and selects the highest predicted risk among the 60 explained records. Undisplayed features are aggregated by the plotting method. Independent masking can break biological correlations and relationships between biomarkers and derived ratios. These are approximate model associations, not causal effects or clinical biomarker validation.

## Verification

Independent verification passed: True. The verification reloads all 12 model and task fits, reproduces all saved validation and test probabilities, recalculates 24 metric rows, checks matching training hashes and confirms SHAP additivity and split membership.

## Reproducibility records

See README.md, requirements.lock.txt, pipeline.py, run_experiment.py, analyze_results.py and verify_results.py. Machine readable evidence is in results/protocol.json, environment.json, validation_checks.json, final_verification.json, comparison_training_manifest.csv, predictions.csv, metrics.csv, shap_settings.json and the saved model and preprocessing files. The figures accompanying this report all come from the run.
