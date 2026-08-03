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


## Diagnostics: mlp / svm_jl (20260801_233756)

**Confusion matrix** (rows=true, cols=pred):
```
[[1975, 751], [708, 1566]]
```

**Classification report:**
```
              precision    recall  f1-score   support

        real       0.74      0.72      0.73      2726
        fake       0.68      0.69      0.68      2274

    accuracy                           0.71      5000
   macro avg       0.71      0.71      0.71      5000
weighted avg       0.71      0.71      0.71      5000

```

**Top 5 hyperparameter configurations (by mean val score):**
- {'clf__alpha': 0.001, 'clf__hidden_layer_sizes': '(100,)'}: 0.6829 (+/- 0.0000)
- {'clf__alpha': 0.0001, 'clf__hidden_layer_sizes': '(100,)'}: 0.6724 (+/- 0.0000)
- {'clf__alpha': 0.0001, 'clf__hidden_layer_sizes': '(64, 32)'}: 0.6635 (+/- 0.0000)
- {'clf__alpha': 0.001, 'clf__hidden_layer_sizes': '(64, 32)'}: 0.6620 (+/- 0.0000)

**Learning curve (train size -> val score):**
- 666: 0.6378
- 1333: 0.6579
- 1999: 0.6632
- 2666: 0.6772
- 3333: 0.6766


## Diagnostics: mlp / fast (20260801_234332)

**Confusion matrix** (rows=true, cols=pred):
```
[[2013, 713], [622, 1652]]
```

**Classification report:**
```
              precision    recall  f1-score   support

        real       0.76      0.74      0.75      2726
        fake       0.70      0.73      0.71      2274

    accuracy                           0.73      5000
   macro avg       0.73      0.73      0.73      5000
weighted avg       0.73      0.73      0.73      5000

```

**Top 5 hyperparameter configurations (by mean val score):**
- {'clf__alpha': 0.001, 'clf__hidden_layer_sizes': '(100,)'}: 0.7286 (+/- 0.0000)
- {'clf__alpha': 0.0001, 'clf__hidden_layer_sizes': '(100,)'}: 0.7285 (+/- 0.0000)
- {'clf__alpha': 0.0001, 'clf__hidden_layer_sizes': '(64, 32)'}: 0.7039 (+/- 0.0000)
- {'clf__alpha': 0.001, 'clf__hidden_layer_sizes': '(64, 32)'}: 0.7025 (+/- 0.0000)

**Learning curve (train size -> val score):**
- 666: 0.6674
- 1333: 0.6888
- 1999: 0.6979
- 2666: 0.7052
- 3333: 0.7078


## Diagnostics: mlp / hq (20260801_235301)

**Confusion matrix** (rows=true, cols=pred):
```
[[2053, 673], [603, 1671]]
```

**Classification report:**
```
              precision    recall  f1-score   support

        real       0.77      0.75      0.76      2726
        fake       0.71      0.73      0.72      2274

    accuracy                           0.74      5000
   macro avg       0.74      0.74      0.74      5000
weighted avg       0.75      0.74      0.75      5000

```

**Top 5 hyperparameter configurations (by mean val score):**
- {'clf__alpha': 0.0001, 'clf__hidden_layer_sizes': '(100,)'}: 0.7335 (+/- 0.0000)
- {'clf__alpha': 0.001, 'clf__hidden_layer_sizes': '(100,)'}: 0.7321 (+/- 0.0000)
- {'clf__alpha': 0.001, 'clf__hidden_layer_sizes': '(64, 32)'}: 0.7048 (+/- 0.0000)
- {'clf__alpha': 0.0001, 'clf__hidden_layer_sizes': '(64, 32)'}: 0.7037 (+/- 0.0000)

**Learning curve (train size -> val score):**
- 666: 0.6057
- 1333: 0.6586
- 1999: 0.6758
- 2666: 0.6983
- 3333: 0.7058


## Diagnostics: mlp / no_denoise (20260801_235802)

**Confusion matrix** (rows=true, cols=pred):
```
[[2018, 708], [623, 1651]]
```

**Classification report:**
```
              precision    recall  f1-score   support

        real       0.76      0.74      0.75      2726
        fake       0.70      0.73      0.71      2274

    accuracy                           0.73      5000
   macro avg       0.73      0.73      0.73      5000
weighted avg       0.73      0.73      0.73      5000

```

**Top 5 hyperparameter configurations (by mean val score):**
- {'clf__alpha': 0.0001, 'clf__hidden_layer_sizes': '(100,)'}: 0.7348 (+/- 0.0000)
- {'clf__alpha': 0.001, 'clf__hidden_layer_sizes': '(100,)'}: 0.7336 (+/- 0.0000)
- {'clf__alpha': 0.001, 'clf__hidden_layer_sizes': '(64, 32)'}: 0.7040 (+/- 0.0000)
- {'clf__alpha': 0.0001, 'clf__hidden_layer_sizes': '(64, 32)'}: 0.7027 (+/- 0.0000)

**Learning curve (train size -> val score):**
- 666: 0.6616
- 1333: 0.6880
- 1999: 0.6929
- 2666: 0.6936
- 3333: 0.7032


## Diagnostics: mlp / embedding_jl (20260802_000324)

**Confusion matrix** (rows=true, cols=pred):
```
[[2041, 685], [643, 1631]]
```

**Classification report:**
```
              precision    recall  f1-score   support

        real       0.76      0.75      0.75      2726
        fake       0.70      0.72      0.71      2274

    accuracy                           0.73      5000
   macro avg       0.73      0.73      0.73      5000
weighted avg       0.73      0.73      0.73      5000

```

**Top 5 hyperparameter configurations (by mean val score):**
- {'clf__alpha': 0.0001, 'clf__hidden_layer_sizes': '(100,)'}: 0.7219 (+/- 0.0000)
- {'clf__alpha': 0.001, 'clf__hidden_layer_sizes': '(100,)'}: 0.7206 (+/- 0.0000)
- {'clf__alpha': 0.0001, 'clf__hidden_layer_sizes': '(64, 32)'}: 0.7022 (+/- 0.0000)
- {'clf__alpha': 0.001, 'clf__hidden_layer_sizes': '(64, 32)'}: 0.6960 (+/- 0.0000)

**Learning curve (train size -> val score):**
- 666: 0.6636
- 1333: 0.6794
- 1999: 0.6907
- 2666: 0.6969
- 3333: 0.7009

