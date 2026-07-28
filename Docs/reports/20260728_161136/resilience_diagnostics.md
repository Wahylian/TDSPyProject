## Diagnostics: svm / svm (20260728_122240)

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


## Diagnostics: svm / svm_jl (20260728_122818)

**Confusion matrix** (rows=true, cols=pred):
```
[[2081, 645], [654, 1620]]
```

**Classification report:**
```
              precision    recall  f1-score   support

        real       0.76      0.76      0.76      2726
        fake       0.72      0.71      0.71      2274

    accuracy                           0.74      5000
   macro avg       0.74      0.74      0.74      5000
weighted avg       0.74      0.74      0.74      5000

```

**Top 5 hyperparameter configurations (by mean val score):**
- {'clf__C': 1.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}: 0.7103 (+/- 0.0000)
- {'clf__C': 1.0, 'clf__gamma': 'auto', 'clf__kernel': 'rbf'}: 0.7103 (+/- 0.0000)
- {'clf__C': 10.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}: 0.6907 (+/- 0.0000)
- {'clf__C': 10.0, 'clf__gamma': 'auto', 'clf__kernel': 'rbf'}: 0.6907 (+/- 0.0000)
- {'clf__C': 0.1, 'clf__gamma': 'scale', 'clf__kernel': 'linear'}: 0.6777 (+/- 0.0000)


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


## Diagnostics: svm / embedding_jl (20260728_123100)

**Confusion matrix** (rows=true, cols=pred):
```
[[2149, 577], [574, 1700]]
```

**Classification report:**
```
              precision    recall  f1-score   support

        real       0.79      0.79      0.79      2726
        fake       0.75      0.75      0.75      2274

    accuracy                           0.77      5000
   macro avg       0.77      0.77      0.77      5000
weighted avg       0.77      0.77      0.77      5000

```

**Top 5 hyperparameter configurations (by mean val score):**
- {'clf__C': 10.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}: 0.7539 (+/- 0.0000)
- {'clf__C': 10.0, 'clf__gamma': 'auto', 'clf__kernel': 'rbf'}: 0.7539 (+/- 0.0000)
- {'clf__C': 1.0, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}: 0.7356 (+/- 0.0000)
- {'clf__C': 1.0, 'clf__gamma': 'auto', 'clf__kernel': 'rbf'}: 0.7356 (+/- 0.0000)
- {'clf__C': 0.1, 'clf__gamma': 'scale', 'clf__kernel': 'rbf'}: 0.7197 (+/- 0.0000)

