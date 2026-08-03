## Diagnostics: hard_svm / svm (20260728_122609)

**Confusion matrix** (rows=true, cols=pred):
```
[[2000, 726], [734, 1540]]
```

**Classification report:**
```
              precision    recall  f1-score   support

        real       0.73      0.73      0.73      2726
        fake       0.68      0.68      0.68      2274

    accuracy                           0.71      5000
   macro avg       0.71      0.71      0.71      5000
weighted avg       0.71      0.71      0.71      5000

```

**Top 5 hyperparameter configurations (by mean val score):**
- {'clf__C': 10000.0}: 0.7147 (+/- 0.0000)
- {'clf__C': 1000000.0}: 0.7147 (+/- 0.0000)


## Diagnostics: hard_svm_kernel / svm (20260722_131023)

**Confusion matrix** (rows=true, cols=pred):
```
[[1197, 1529], [573, 1701]]
```

**Classification report:**
```
              precision    recall  f1-score   support

        real       0.68      0.44      0.53      2726
        fake       0.53      0.75      0.62      2274

    accuracy                           0.58      5000
   macro avg       0.60      0.59      0.58      5000
weighted avg       0.61      0.58      0.57      5000

```

**Top 5 hyperparameter configurations (by mean val score):**
- {'clf__C': 10000.0}: 0.5796 (+/- 0.0000)
- {'clf__C': 1000000.0}: 0.5796 (+/- 0.0000)


## Diagnostics: hgb / svm (20260728_122521)

**Confusion matrix** (rows=true, cols=pred):
```
[[2063, 663], [713, 1561]]
```

**Classification report:**
```
              precision    recall  f1-score   support

        real       0.74      0.76      0.75      2726
        fake       0.70      0.69      0.69      2274

    accuracy                           0.72      5000
   macro avg       0.72      0.72      0.72      5000
weighted avg       0.72      0.72      0.72      5000

```

**Top 5 hyperparameter configurations (by mean val score):**
- {'clf__learning_rate': 0.05, 'clf__max_iter': 200}: 0.7168 (+/- 0.0000)
- {'clf__learning_rate': 0.1, 'clf__max_iter': 200}: 0.7154 (+/- 0.0000)
- {'clf__learning_rate': 0.05, 'clf__max_iter': 100}: 0.7109 (+/- 0.0000)
- {'clf__learning_rate': 0.1, 'clf__max_iter': 100}: 0.7086 (+/- 0.0000)


## Diagnostics: linreg / svm (20260728_122448)

**Confusion matrix** (rows=true, cols=pred):
```
[[1989, 737], [731, 1543]]
```

**Classification report:**
```
              precision    recall  f1-score   support

        real       0.73      0.73      0.73      2726
        fake       0.68      0.68      0.68      2274

    accuracy                           0.71      5000
   macro avg       0.70      0.70      0.70      5000
weighted avg       0.71      0.71      0.71      5000

```

**Top 5 hyperparameter configurations (by mean val score):**
- {'clf__fit_intercept': True}: 0.7163 (+/- 0.0000)
- {'clf__fit_intercept': False}: 0.0939 (+/- 0.0000)


## Diagnostics: logreg / svm (20260728_122424)

**Confusion matrix** (rows=true, cols=pred):
```
[[2008, 718], [732, 1542]]
```

**Classification report:**
```
              precision    recall  f1-score   support

        real       0.73      0.74      0.73      2726
        fake       0.68      0.68      0.68      2274

    accuracy                           0.71      5000
   macro avg       0.71      0.71      0.71      5000
weighted avg       0.71      0.71      0.71      5000

```

**Top 5 hyperparameter configurations (by mean val score):**
- {'clf__C': 0.1}: 0.7113 (+/- 0.0000)
- {'clf__C': 1.0}: 0.7109 (+/- 0.0000)
- {'clf__C': 10.0}: 0.7109 (+/- 0.0000)


## Diagnostics: mlp / svm (20260728_122541)

**Confusion matrix** (rows=true, cols=pred):
```
[[1990, 736], [618, 1656]]
```

**Classification report:**
```
              precision    recall  f1-score   support

        real       0.76      0.73      0.75      2726
        fake       0.69      0.73      0.71      2274

    accuracy                           0.73      5000
   macro avg       0.73      0.73      0.73      5000
weighted avg       0.73      0.73      0.73      5000

```

**Top 5 hyperparameter configurations (by mean val score):**
- {'clf__alpha': 0.0001, 'clf__hidden_layer_sizes': '(100,)'}: 0.7263 (+/- 0.0000)
- {'clf__alpha': 0.001, 'clf__hidden_layer_sizes': '(100,)'}: 0.7245 (+/- 0.0000)
- {'clf__alpha': 0.001, 'clf__hidden_layer_sizes': '(64, 32)'}: 0.6990 (+/- 0.0000)
- {'clf__alpha': 0.0001, 'clf__hidden_layer_sizes': '(64, 32)'}: 0.6945 (+/- 0.0000)

**Learning curve (train size -> val score):**
- 666: 0.6579
- 1333: 0.6899
- 1999: 0.6990
- 2666: 0.7112
- 3333: 0.7044


## Diagnostics: rf / svm (20260728_122502)

**Confusion matrix** (rows=true, cols=pred):
```
[[2219, 507], [894, 1380]]
```

**Classification report:**
```
              precision    recall  f1-score   support

        real       0.71      0.81      0.76      2726
        fake       0.73      0.61      0.66      2274

    accuracy                           0.72      5000
   macro avg       0.72      0.71      0.71      5000
weighted avg       0.72      0.72      0.72      5000

```

**Top 10 feature importances:**
- feature[4]: 0.0477
- feature[10]: 0.0273
- feature[21]: 0.0153
- feature[9]: 0.0113
- feature[7]: 0.0109
- feature[29]: 0.0105
- feature[14]: 0.0105
- feature[32]: 0.0105
- feature[19]: 0.0103
- feature[16]: 0.0103

**OOB score:** 0.7183

**Top 5 hyperparameter configurations (by mean val score):**
- {'clf__max_depth': 20, 'clf__n_estimators': 400}: 0.6736 (+/- 0.0000)
- {'clf__max_depth': 20, 'clf__n_estimators': 200}: 0.6594 (+/- 0.0000)
- {'clf__max_depth': None, 'clf__n_estimators': 400}: 0.6568 (+/- 0.0000)
- {'clf__max_depth': None, 'clf__n_estimators': 200}: 0.6512 (+/- 0.0000)


## Diagnostics: ridge / svm (20260728_122436)

**Confusion matrix** (rows=true, cols=pred):
```
[[1989, 737], [731, 1543]]
```

**Classification report:**
```
              precision    recall  f1-score   support

        real       0.73      0.73      0.73      2726
        fake       0.68      0.68      0.68      2274

    accuracy                           0.71      5000
   macro avg       0.70      0.70      0.70      5000
weighted avg       0.71      0.71      0.71      5000

```

**Top 5 hyperparameter configurations (by mean val score):**
- {'clf__alpha': 10.0}: 0.7167 (+/- 0.0000)
- {'clf__alpha': 0.1}: 0.7163 (+/- 0.0000)
- {'clf__alpha': 1.0}: 0.7163 (+/- 0.0000)


## Diagnostics: svm / svm (20260722_131056)

**Confusion matrix** (rows=true, cols=pred):
```
[[2114, 612], [632, 1642]]
```

**Classification report:**
```
              precision    recall  f1-score   support

        real       0.77      0.78      0.77      2726
        fake       0.73      0.72      0.73      2274

    accuracy                           0.75      5000
   macro avg       0.75      0.75      0.75      5000
weighted avg       0.75      0.75      0.75      5000

```

**Top 5 hyperparameter configurations (by mean val score):**
- {'clf__C': 1.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}: 0.7470 (+/- 0.0000)
- {'clf__C': 1.0, 'clf__gamma': 'auto', 'clf__kernel': 'rbf'}: 0.7470 (+/- 0.0000)
- {'clf__C': 10.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}: 0.7358 (+/- 0.0000)
- {'clf__C': 10.0, 'clf__gamma': 'auto', 'clf__kernel': 'rbf'}: 0.7358 (+/- 0.0000)
- {'clf__C': 0.1, 'clf__gamma': 'scale', 'clf__kernel': 'linear'}: 0.7099 (+/- 0.0000)

