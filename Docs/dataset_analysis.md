# Dataset Analysis & Pre-Processing Notes
### Deepfake Detection Project — TDS

---

## 1. Dataset Overview

**Source:** `prithivsakthiur/deepfake-vs-real-20k` (Kaggle) — downloaded locally
as `.jpg` image files.

| Property | Value |
|---|---|
| Total images | 14,433 |
| Layout | Two class folders of `.jpg` images: `Real/` and `Deepfake/` |
| Task | Binary image classification (Real vs Deepfake) |
| Label source | Parent folder name (`Real` → 0, `Deepfake` → 1) |

Unlike a tabular dataset, this one ships purely as images on disk — there is no
accompanying metadata CSV. `create_split.py` constructs our own manifest from the
folder structure (see Section 5), giving full control over the split ratios and
random seed so all results are reproducible and directly comparable across models.

**Split strategy: seeded random split (70/15/15)**

`create_split.py` shuffles all images with a fixed seed (default 42) and
partitions them 70% / 15% / 15% into train / val / test. The split is a plain
random shuffle (not explicitly stratified); because shuffling preserves the
overall class ratio closely, each partition still lands near the global
~34% Real / ~66% Deepfake balance:

| Split | Proportion | Real (0) | Deepfake (1) | Total |
|---|---|---|---|---|
| Train | 70% | 3,457 | 6,646 | 10,103 |
| Validation | 15% | 674 | 1,490 | 2,164 |
| Test | 15% | 726 | 1,440 | 2,166 |
| **Total** | 100% | 4,857 | 9,576 | 14,433 |

---

## 2. Class Imbalance

**Finding:** The dataset is imbalanced — 9,576 Deepfake (66.3%) vs 4,857 Real
(33.7%), a ratio of approximately 1.97:1 in favour of Deepfake.

**Decision: No resampling (monitor instead).** Both classes are well-represented
in absolute terms (thousands of images each), so the data is left untouched:

- Pass `class_weight='balanced'` to classical models (SVM, LR) to adjust the loss
  function without altering the data.
- Monitor per-class recall and F1 in every evaluation. If a model is
  systematically biased toward predicting Deepfake as a shortcut, resampling will
  be revisited — specifically **undersampling the Deepfake class** (dropping
  ~4,700 rows to balance the classes) rather than oversampling, to avoid
  synthetic-sample artifacts.

---

## 3. Labels & Manifest Schema

This dataset carries **no per-image metadata** — the only signal is the image
content itself, and the only label is the class folder an image lives in. There
are therefore no leakage columns or auxiliary attributes to audit (as there would
be for a tabular dataset). The label is assigned directly from the class folder:

| Folder | Label |
|---|---|
| `Real/` | `0` |
| `Deepfake/` | `1` |

`create_split.py` records this in a manifest CSV, `datasets/dataset_split.csv`,
which is the single source of truth consumed downstream:

| Column | Description |
|---|---|
| `photo_name` | The image filename (e.g. `CCO (410).jpg`) |
| `photo_path` | The image path relative to the project root, forward-slashed |
| `label` | Integer class label: `0` for Real, `1` for Deepfake |
| `split` | The assigned partition: `train`, `val`, or `test` |

Because the label is derived from folder membership rather than any recorded
attribute, there is no risk of metadata leakage: every model must learn from the
image pixels alone.

---

## 4. Image Pre-Processing Pipeline

Images are streamed off disk by `extract_features.py` as decoded BGR `uint8`
arrays (OpenCV convention) and fed into the shared `preprocessing` package, which
supplies the resize / grayscale / normalize / denoise / vectorize / reduce
building blocks (documented in `IMAGE_PREPROCESSING_GUIDE.md`).

For the deep models, inputs are resized to 224×224×3 and normalised with ImageNet
statistics, which is appropriate since all deep models are initialised from
ImageNet-pretrained weights:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

**Data augmentation** (training set only, not validation/test):
- Random horizontal flip
- Small rotation (±10°)
- Slight colour jitter (brightness/contrast ±0.2)

With ~10,100 training images, augmentation still helps the deep models
generalise. It is applied at the DataLoader level, not inside the streaming loader.

---

## 5. Existing Codebase

The data pipeline runs in three ordered stages, each a standalone script. The
images ship as local `.jpg` files on disk, so model training reads them straight
from storage — there is no per-image URL fetching at training time.

### `download_dataset.py`

A minimal script that downloads the dataset from Kaggle using the `kagglehub`
library and copies it into the project's `datasets/` folder, keeping everything
self-contained.

```python
import shutil
from pathlib import Path
import kagglehub

path = kagglehub.dataset_download("prithivsakthiur/deepfake-vs-real-20k")
dest = Path(__file__).parent / "datasets" / "deepfake-vs-real-20k"
shutil.copytree(path, dest, dirs_exist_ok=True)
```

This lands the images under
`datasets/deepfake-vs-real-20k/Deep-vs-Real/`, which holds two class subfolders
of `.jpg` images: `Real/` and `Deepfake/`.

### `create_split.py`

Scans the dataset directory and builds a deterministic 70/15/15 train/val/test
split. It walks the `Real/` and `Deepfake/` subfolders, records every `.jpg`
image, shuffles the rows with a fixed seed (default `42`), partitions them
70% / 15% / 15% into train / val / test, and writes
`datasets/dataset_split.csv` with these columns:

| Column | Description |
|---|---|
| `photo_name` | The image filename (e.g. `CCO (410).jpg`) |
| `photo_path` | The image path relative to the project root, using forward slashes |
| `label` | Integer class label: `0` for Real, `1` for Deepfake |
| `split` | The assigned partition: `train`, `val`, or `test` |

The split sizes are `int(0.70 * n)` train, `int(0.15 * n)` val, and the remainder
test. Re-running with the same seed reproduces an identical manifest, so all
results stay comparable across models. The seed is configurable from the command
line:

```bash
python create_split.py              # default seed 42
python create_split.py --seed 7     # a different shuffle
```

### `extract_features.py`

A streaming data loader that yields one `(image, label)` pair at a time so the
full set of images is never held in memory at once. Key design points:

- **Generator-based (`get_feature_stream`):** Reads `datasets/dataset_split.csv`,
  filters it to the requested split, loads each image **from its local path**,
  decodes it to a BGR NumPy array (OpenCV convention, matching the
  `preprocessing` package), and yields it with its integer label.
- **Randomized order:** The lightweight `(path, label)` list is shuffled up front
  (seeded for reproducibility) so the stream is not biased by file or class
  order; only the cheap path/label strings are held in memory.
- **Resilient:** A path that is missing or undecodable is skipped (with a
  warning) rather than aborting the stream.

Each item yielded is a `(np.ndarray, int)` pair, where the array is a decoded
BGR image of shape `(height, width, 3)` and dtype `uint8`, ready to feed into the
`preprocessing` pipeline (Section 4).

---

## 6. Planned Models & Feature Strategies

The project intends to systematically explore all combinations across the following axes, rather than committing to a single pipeline upfront. The goal is to understand which components drive performance and why.

### 6.1 Classical Models

| Model | Notes |
|---|---|
| Hard-SVM | No slack — assumes clean linearly separable data. Will test on high-quality feature representations only. |
| Soft-SVM | C-parameter controls margin softness. Will sweep C values. |
| Logistic Regression | Strong linear baseline, probabilistic outputs, fast to train. |
| Linear Regression | Included for completeness; not ideal for classification but will be evaluated with a threshold. |

All classical models require a fixed-length feature vector. Two feature extraction strategies will be tried for each:

**Strategy A — Flattened pixels:** Resize to 64×64, flatten to 12,288-dim vector. Computationally cheap but high-dimensional and noisy.

**Strategy B — Pretrained CNN embeddings:** Pass images through a frozen ResNet-50 (or EfficientNet-B0) and extract the penultimate layer (~2048 dims). Far more semantically rich.

### 6.2 Dimensionality Reduction (for classical models)

Applied after feature extraction, before the classifier. Both options will be tested:

| Method | Rationale |
|---|---|
| **PCA** | Finds directions of maximum variance. Best for reducing to a small number of components (e.g. 50–200). Data-driven. |
| **Johnson-Lindenstrauss (JL) random projection** | Preserves pairwise Euclidean distances via random matrix multiplication. No fitting required — fully data-independent. Theoretically well-suited to SVMs (distance-based models). Target dims ~200–400 for this dataset size. |

With ~10,100 training samples, JL targets around 200–400 dimensions for reasonable distortion tolerance (ε ≈ 0.5–0.9). PCA can be pushed further (50–128 components) while retaining most variance. Both will be evaluated.

Scikit-learn implementations: `sklearn.decomposition.PCA` and `sklearn.random_projection.SparseRandomProjection`.

### 6.3 Deep Learning Models

| Model | Strategy |
|---|---|
| **CNN (custom)** | Lightweight conv architecture trained from scratch — mainly as a lower-bound baseline. |
| **ResNet-50 (fine-tuned)** | Pretrained on ImageNet, fine-tune full network. Strong standard baseline. |
| **EfficientNet-B0/B3** | Better accuracy-per-parameter ratio. Will fine-tune. |
| **Vision Transformer (ViT-Base/16)** | Patch-based attention model. Pretrained from `google/vit-base-patch16-224`. Will fine-tune. |

Input: 224×224×3 normalised images for all deep models.

### 6.4 Additional Models

| Model | Rationale |
|---|---|
| **XGBoost / LightGBM** | Gradient boosted trees on CNN embeddings. Often competitive with SVM at lower tuning cost. |
| **Frequency-domain SVM** | Extract DCT/FFT power spectrum from images before feeding to SVM. GAN-based generators are known to leave spectral artifacts — this is a domain-informed approach. |
| **MLP on CNN embeddings** | Simple fully-connected classifier on top of pretrained features. Bridges classical and deep approaches. |

---

## 7. Full Combination Matrix

The following combinations will be explored:

```
Feature Source      × Dimensionality Reduction × Classifier
─────────────────────────────────────────────────────────────
Raw pixels (64×64)  × None / PCA / JL           × Hard-SVM / Soft-SVM / LR / LinReg
CNN embeddings      × None / PCA / JL           × Hard-SVM / Soft-SVM / LR / LinReg
CNN embeddings      × None                      × XGBoost / LightGBM / MLP
FFT spectrum        × None / PCA                × Soft-SVM / LR
Raw images (224×224)× Augmentation              × Custom CNN / ResNet / EfficientNet / ViT
```

All configurations will be evaluated on the same held-out test split using accuracy, per-class precision/recall, F1-score, and AUC-ROC.

---

## 8. Predictions: Expected Best-Performing Combinations

These are informed predictions based on the nature of the task (GAN-generated image detection) and the dataset characteristics, made prior to running experiments.

### 🥇 Expected Top Performer: Fine-tuned ViT or EfficientNet

Vision Transformers have shown strong results on deepfake detection benchmarks because self-attention can capture long-range spatial inconsistencies that GANs often introduce — particularly in background/hair regions that generative models can render with subtle artifacts. EfficientNet is the backup if training budget is limited. Both benefit from the ImageNet pretraining.

### 🥈 Expected Second Tier: CNN Embeddings + Soft-SVM or XGBoost

A pretrained CNN (ResNet/EfficientNet) used purely as a feature extractor, with a Soft-SVM or gradient boosted tree on top, is a well-established and reliable pattern. With ~10,100 training images, the frozen-backbone approach remains attractive since it needs far less data than full fine-tuning. This combination should outperform end-to-end fine-tuning if training data is too small to move the pretrained weights usefully.

### 🥉 Expected Third Tier: Frequency-domain features + Soft-SVM

GAN-based generators are known to produce characteristic high-frequency artifacts in the DCT/FFT domain — something real camera images do not exhibit. A Soft-SVM on a frequency power spectrum is therefore a strong domain-informed baseline that may punch well above its computational weight. This is a particularly interesting result to report because it would demonstrate that the deepfakes are detectable without any deep learning at all.

### Predicted Weakest Configurations

- **Hard-SVM on raw pixels:** The hard margin assumption will almost certainly be violated with high-dimensional noisy pixel features. Expected to fail or require extremely aggressive dimensionality reduction.
- **Linear Regression with threshold:** Fundamental mismatch between the model's output space and the binary task. Included for completeness and as a pedagogical comparison point, not for performance.
- **Custom CNN trained from scratch:** With ~10,100 images and no pretraining, it is still expected to be clearly dominated by the ImageNet-pretrained alternatives.
- **PCA/JL + Hard-SVM on raw pixels:** Even with dimensionality reduction, the hard-margin assumption is fragile. JL may help slightly more than PCA here (distance preservation), but neither will rescue Hard-SVM on raw pixels.

### The Most Interesting Predicted Finding

The **JL embedding vs PCA** comparison on Soft-SVM is expected to produce very similar results — JL's distance-preservation guarantee makes it theoretically equivalent for SVMs, but PCA's data-awareness may give it a slight edge at aggressive compression ratios. If JL matches PCA at, say, 300 dims, this is a practically useful result: JL requires no fitting and can be applied to new data instantly, making it preferable in a pipeline where the feature distribution may shift.

---

## 9. Open Questions for the Report

- Does data augmentation help the deep models, or does it introduce too much variance given the dataset size?
- Can the frequency-domain approach replicate or approach the accuracy of deep models, suggesting the artifacts are fundamentally spectral in nature?
- Does the choice of PCA vs JL matter more for Hard-SVM (where margins are strict) than for Soft-SVM?

---

*Document generated after exploratory data analysis. All predictions are pre-experiment hypotheses and will be revisited in the final report.*