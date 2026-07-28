## Diagnostics: hard_svm / embedding_pca (20260728_131015)

**Confusion matrix** (rows=true, cols=pred):
```
[[2137, 589], [466, 1808]]
```

**Classification report:**
```
              precision    recall  f1-score   support

        real       0.82      0.78      0.80      2726
        fake       0.75      0.80      0.77      2274

    accuracy                           0.79      5000
   macro avg       0.79      0.79      0.79      5000
weighted avg       0.79      0.79      0.79      5000

```

**Top 5 hyperparameter configurations (by mean val score):**
- {'clf__C': 10000.0}: 0.7840 (+/- 0.0000)
- {'clf__C': 1000000.0}: 0.7840 (+/- 0.0000)

**Learning curve (train size -> val score):**
- 666: 0.7028
- 1333: 0.7499
- 1999: 0.7597
- 2666: 0.7725
- 3333: 0.7780


## Diagnostics: hard_svm_kernel / embedding_pca (20260728_131035)

**Confusion matrix** (rows=true, cols=pred):
```
[[1642, 1084], [668, 1606]]
```

**Classification report:**
```
              precision    recall  f1-score   support

        real       0.71      0.60      0.65      2726
        fake       0.60      0.71      0.65      2274

    accuracy                           0.65      5000
   macro avg       0.65      0.65      0.65      5000
weighted avg       0.66      0.65      0.65      5000

```

**Top 5 hyperparameter configurations (by mean val score):**
- {'clf__C': 10000.0}: 0.6755 (+/- 0.0000)
- {'clf__C': 1000000.0}: 0.6755 (+/- 0.0000)

**Learning curve (train size -> val score):**
- 666: 0.6593
- 1333: 0.6283
- 1999: 0.6287
- 2666: 0.6380
- 3333: 0.6366


## Diagnostics: hgb / embedding_pca (20260728_130753)

**Confusion matrix** (rows=true, cols=pred):
```
[[2137, 589], [448, 1826]]
```

**Classification report:**
```
              precision    recall  f1-score   support

        real       0.83      0.78      0.80      2726
        fake       0.76      0.80      0.78      2274

    accuracy                           0.79      5000
   macro avg       0.79      0.79      0.79      5000
weighted avg       0.79      0.79      0.79      5000

```

**Top 5 hyperparameter configurations (by mean val score):**
- {'clf__learning_rate': 0.1, 'clf__max_iter': 200}: 0.7862 (+/- 0.0000)
- {'clf__learning_rate': 0.1, 'clf__max_iter': 100}: 0.7782 (+/- 0.0000)
- {'clf__learning_rate': 0.05, 'clf__max_iter': 200}: 0.7722 (+/- 0.0000)
- {'clf__learning_rate': 0.05, 'clf__max_iter': 100}: 0.7699 (+/- 0.0000)

**Learning curve (train size -> val score):**
- 666: 0.6931
- 1333: 0.7363
- 1999: 0.7458
- 2666: 0.7574
- 3333: 0.7570


## Diagnostics: linreg / embedding_pca (20260728_130949)

**Confusion matrix** (rows=true, cols=pred):
```
[[2118, 608], [477, 1797]]
```

**Classification report:**
```
              precision    recall  f1-score   support

        real       0.82      0.78      0.80      2726
        fake       0.75      0.79      0.77      2274

    accuracy                           0.78      5000
   macro avg       0.78      0.78      0.78      5000
weighted avg       0.78      0.78      0.78      5000

```

**Top 5 hyperparameter configurations (by mean val score):**
- {'clf__fit_intercept': True}: 0.7807 (+/- 0.0000)
- {'clf__fit_intercept': False}: 0.1694 (+/- 0.0000)

**Learning curve (train size -> val score):**
- 666: 0.7201
- 1333: 0.7554
- 1999: 0.7612
- 2666: 0.7727
- 3333: 0.7776


## Diagnostics: logreg / embedding_pca (20260728_130732)

**Confusion matrix** (rows=true, cols=pred):
```
[[2140, 586], [474, 1800]]
```

**Classification report:**
```
              precision    recall  f1-score   support

        real       0.82      0.79      0.80      2726
        fake       0.75      0.79      0.77      2274

    accuracy                           0.79      5000
   macro avg       0.79      0.79      0.79      5000
weighted avg       0.79      0.79      0.79      5000

```

**Top 5 hyperparameter configurations (by mean val score):**
- {'clf__C': 1.0}: 0.7872 (+/- 0.0000)
- {'clf__C': 10.0}: 0.7872 (+/- 0.0000)
- {'clf__C': 0.1}: 0.7846 (+/- 0.0000)

**Learning curve (train size -> val score):**
- 666: 0.7083
- 1333: 0.7508
- 1999: 0.7613
- 2666: 0.7718
- 3333: 0.7751


## Diagnostics: mlp / embedding_pca (20260728_130830)

**Confusion matrix** (rows=true, cols=pred):
```
[[2176, 550], [400, 1874]]
```

**Classification report:**
```
              precision    recall  f1-score   support

        real       0.84      0.80      0.82      2726
        fake       0.77      0.82      0.80      2274

    accuracy                           0.81      5000
   macro avg       0.81      0.81      0.81      5000
weighted avg       0.81      0.81      0.81      5000

```

**Top 5 hyperparameter configurations (by mean val score):**
- {'clf__alpha': 0.0001, 'clf__hidden_layer_sizes': '(100,)'}: 0.8029 (+/- 0.0000)
- {'clf__alpha': 0.001, 'clf__hidden_layer_sizes': '(100,)'}: 0.8011 (+/- 0.0000)
- {'clf__alpha': 0.001, 'clf__hidden_layer_sizes': '(64, 32)'}: 0.7809 (+/- 0.0000)
- {'clf__alpha': 0.0001, 'clf__hidden_layer_sizes': '(64, 32)'}: 0.7806 (+/- 0.0000)

**Learning curve (train size -> val score):**
- 666: 0.7062
- 1333: 0.7507
- 1999: 0.7610
- 2666: 0.7740
- 3333: 0.7847


## Diagnostics: rf / embedding_pca (20260728_130859)

**Confusion matrix** (rows=true, cols=pred):
```
[[2117, 609], [639, 1635]]
```

**Classification report:**
```
              precision    recall  f1-score   support

        real       0.77      0.78      0.77      2726
        fake       0.73      0.72      0.72      2274

    accuracy                           0.75      5000
   macro avg       0.75      0.75      0.75      5000
weighted avg       0.75      0.75      0.75      5000

```

**Top 10 feature importances:**
- feature[1]: 0.1114
- feature[6]: 0.0162
- feature[0]: 0.0143
- feature[2]: 0.0137
- feature[17]: 0.0125
- feature[5]: 0.0109
- feature[52]: 0.0105
- feature[12]: 0.0103
- feature[67]: 0.0100
- feature[7]: 0.0089

**OOB score:** 0.7540

**Top 5 hyperparameter configurations (by mean val score):**
- {'clf__max_depth': 20, 'clf__n_estimators': 200}: 0.7511 (+/- 0.0000)
- {'clf__max_depth': None, 'clf__n_estimators': 400}: 0.7453 (+/- 0.0000)
- {'clf__max_depth': 20, 'clf__n_estimators': 400}: 0.7444 (+/- 0.0000)
- {'clf__max_depth': None, 'clf__n_estimators': 200}: 0.7413 (+/- 0.0000)

**Learning curve (train size -> val score):**
- 666: 0.6297
- 1333: 0.6804
- 1999: 0.6905
- 2666: 0.7003
- 3333: 0.7069


## Diagnostics: ridge / embedding_pca (20260728_130929)

**Confusion matrix** (rows=true, cols=pred):
```
[[2118, 608], [477, 1797]]
```

**Classification report:**
```
              precision    recall  f1-score   support

        real       0.82      0.78      0.80      2726
        fake       0.75      0.79      0.77      2274

    accuracy                           0.78      5000
   macro avg       0.78      0.78      0.78      5000
weighted avg       0.78      0.78      0.78      5000

```

**Top 5 hyperparameter configurations (by mean val score):**
- {'clf__alpha': 0.1}: 0.7807 (+/- 0.0000)
- {'clf__alpha': 1.0}: 0.7807 (+/- 0.0000)
- {'clf__alpha': 10.0}: 0.7807 (+/- 0.0000)

**Learning curve (train size -> val score):**
- 666: 0.7183
- 1333: 0.7530
- 1999: 0.7609
- 2666: 0.7743
- 3333: 0.7764


## Diagnostics: svm / embedding_pca (20260728_122943)

**Confusion matrix** (rows=true, cols=pred):
```
[[2240, 486], [365, 1909]]
```

**Classification report:**
```
              precision    recall  f1-score   support

        real       0.86      0.82      0.84      2726
        fake       0.80      0.84      0.82      2274

    accuracy                           0.83      5000
   macro avg       0.83      0.83      0.83      5000
weighted avg       0.83      0.83      0.83      5000

```

**Top 5 hyperparameter configurations (by mean val score):**
- {'clf__C': 10.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}: 0.8165 (+/- 0.0000)
- {'clf__C': 10.0, 'clf__gamma': 'auto', 'clf__kernel': 'rbf'}: 0.8165 (+/- 0.0000)
- {'clf__C': 1.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}: 0.8140 (+/- 0.0000)
- {'clf__C': 1.0, 'clf__gamma': 'auto', 'clf__kernel': 'rbf'}: 0.8140 (+/- 0.0000)
- {'clf__C': 10.0, 'clf__gamma': 'scale', 'clf__kernel': 'linear'}: 0.7825 (+/- 0.0000)

