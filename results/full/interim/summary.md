# PRELIMINARY: logistic regression baseline, random split

Interim results, not final. Full data after cleaning (28,572 prompts), cheap = `mistralai/mixtral-8x7b-chat`, strong = `gpt-4-1106-preview`. Random 70/15/15 split stratified by benchmark; 5 seeds (0, 1, 2, 3, 4); `C` chosen on validation curve AUC; every number is measured on the test split, as mean ± standard deviation over seeds. Source: `metrics_random.csv` and `reference_policies.csv` in this folder.

| Router | Test AUC | Lift over random | Cost@95%, test curve (% of always-strong) | Cost at validation threshold (%) | Met 95% target | ROC AUC | Chosen C |
|---|---|---|---|---|---|---|---|
| LR, TF-IDF | 0.6961 ± 0.0042 | 0.0135 ± 0.0036 | 76.9 ± 1.4 | 76.6 ± 1.3 | 2 of 5 seeds | 0.7236 ± 0.0103 | 0.01, 10, 0.01, 0.01, 10 |
| LR, hand-built | 0.6933 ± 0.0037 | 0.0107 ± 0.0044 | 81.0 ± 1.9 | 79.6 ± 1.8 | 1 of 5 seeds | 0.6980 ± 0.0077 | 10, 10, 10, 10, 10 |

Reference points on the same test splits (quality = mean score of the chosen model):

| Policy | Quality | Cost (% of always-strong) |
|---|---|---|
| always-cheap | 0.5486 ± 0.0041 | 5.3 ± 0.1 |
| always-strong | 0.8167 ± 0.0046 | 100.0 ± 0.0 |
| random | 0.6826 ± 0.0041 | 52.7 ± 0.0 |
| oracle | 0.8571 ± 0.0049 | 36.8 ± 1.0 |

Random routing's AUC (the baseline that lift is measured from): 0.6826 ± 0.0041.

Runtime 33 s. Embeddings, the other routers and leave-one-benchmark-out are not run yet.
