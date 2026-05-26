# Dataset Analysis & Pre-Processing Notes
### Deepfake Detection Project — TDS

---

## 1. Dataset Overview

**Source:** `FINAL_DATASET.csv` — Chuneeb Deepfake Detection Dataset 2026 (v2.0)

| Property | Value |
|---|---|
| Total samples | 6,557 |
| Columns | 17 |
| Task | Binary image classification (REAL vs FAKE) |
| Image source | URLs (Unsplash for real, MultiSource/MultiAvatar for fake) |
| Fake generation method | StyleGAN3 |
| Collection period | 2026 (single year) |

The CSV contains a pre-existing `dataset_split` column (train/val/test), but we have decided **not** to use it. We will construct our own splits from scratch to have full control over the split ratios, stratification, and random seed — ensuring all results are reproducible and directly comparable across models.

**Planned split strategy: stratified random split**

A stratified split preserves the class ratio (57.4% FAKE / 42.6% REAL) in each partition, which is important for fair evaluation given the mild imbalance.

| Split | Proportion | Approx. FAKE | Approx. REAL | Approx. Total |
|---|---|---|---|---|
| Train | 70% | 2,637 | 1,953 | 4,590 |
| Validation | 15% | 565 | 419 | 984 |
| Test | 15% | 565 | 418 | 983 |

The `dataset_split` column will be **dropped** along with the leakage columns listed in Section 3.

---

## 2. Class Imbalance

**Finding:** The dataset is moderately imbalanced — 3,767 FAKE (57.4%) vs 2,790 REAL (42.6%), a ratio of approximately 1.35:1.

**Decision: No resampling.** This imbalance is mild and both classes are well-represented in absolute terms (thousands of samples each). Aggressive resampling at this ratio risks introducing more problems than it solves. The approach taken is:

- Pass `class_weight='balanced'` to classical models (SVM, LR) to adjust the loss function without touching the data.
- Monitor per-class recall and F1 in all evaluations. If a model is systematically biased toward predicting FAKE as a shortcut, resampling will be revisited — specifically **undersampling the FAKE class** (dropping ~977 rows) rather than oversampling, to avoid synthetic sample artifacts.

---

## 3. Column Audit — Leakage, Labels & Usable Attributes

The 17 CSV columns fall into four categories. Only the final category produces actual model inputs.

### Target variable (the label)

| Column | Role |
|---|---|
| `label` | Primary target - "REAL" or "FAKE" - dropped as redundant |
| `label_numeric` | Numeric encoding of `label` - will be used as the actual label |

### Leakage columns — drop before any training

During exploratory analysis, several columns were found to be **perfectly correlated with the label**. These are metadata that was recorded *after* the fact (e.g. which source the image came from, how the fake was generated) and would not be available in a real deployment. Leaving them in would allow any model to achieve near-100% accuracy trivially, making results meaningless.

| Column | Leakage Pattern |
|---|---|
| `source` | MultiSource = FAKE, Unsplash = REAL (100%) |
| `resolution` | 1024×1024 = FAKE, 1080×1080 = REAL (100%) |
| `detection_difficulty` | Easy = REAL, Hard/Medium = FAKE (100%) |
| `fake_method` | NaN = REAL, "StyleGAN3" = FAKE (100%) |
| `category` | "AI Generated" = FAKE, "Authentic" = REAL (direct label copy) |

### Administrative / zero-information columns — drop

These columns carry no signal and are not usable as inputs:

| Column | Reason |
|---|---|
| `image_id` | Sequential row index — encodes nothing |
| `version` | Constant "v2.0" across all rows |
| `year` | Constant 2026 across all rows |
| `dataset_split` | Replaced by our own custom split (see Section 1) |

### Usable attributes — candidates for model input

These columns survived the audit with no direct correlation to the label. They will be selectively included depending on the model type:

| Column | Type | Notes |
|---|---|---|
| `image_url` | String | The primary input — the actual image is fetched from this URL |
| `confidence_score` | Float [0.8, 0.99] | FAKE mean 0.89, REAL mean 0.92 — distributions overlap, not a giveaway |
| `gender` | Categorical | Unknown / Female / Male — no correlation with label |
| `age_group` | Categorical | 18–25 / 26–35 / 36–50 / 50+ — balanced across classes |
| `image_quality` | Categorical | High / Medium — appears in both classes |

Of these, `image_url` is the core input for all models. The remaining four are metadata attributes that may be appended to classical model feature vectors as an experiment, but are not expected to add much signal given the image content dominates.

---

## 4. Image Pre-Processing Pipeline

All image-based models share a common base pipeline, implemented in the `preprocess_image` function in `extractfeatures.py` (currently a placeholder):

```python
from PIL import Image
import io, numpy as np

def preprocess_image(image_bytes: bytes, size=(224, 224)) -> np.ndarray:
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize(size)
    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    return (arr - mean) / std  # shape: (224, 224, 3)
```

Normalization uses ImageNet statistics, which are appropriate since all deep models will be initialised from ImageNet-pretrained weights.

**Data augmentation** (training set only, not validation/test):
- Random horizontal flip
- Small rotation (±10°)
- Slight colour jitter (brightness/contrast ±0.2)

With only ~4,500 training images this augmentation is important for deep models to generalise. Augmentation is applied at the DataLoader level, not in `preprocess_image`.

---

## 5. Existing Codebase

Two scripts have already been written as the foundation of the data pipeline.

### `downloaddataset.py`

A minimal script that downloads the dataset from Kaggle using the `kagglehub` library. It overrides the default cache location to store the dataset relative to the project directory, keeping everything self-contained.

```python
import kagglehub, os
os.environ["KAGGLEHUB_CACHE"] = os.path.dirname(os.path.abspath(__file__))
path = kagglehub.dataset_download("chuneeb/deepfake-detection-dataset-2026")
```

This ensures the CSV lands at `datasets/chuneeb/deepfake-detection-dataset-2026/versions/1/FINAL_DATASET.csv`.

### `extractfeatures.py`

A more substantial script that implements a streaming data pipeline combining image fetching with network traffic capture. Key design decisions already made:

- **Generator-based (`get_data_stream`):** Yields one record at a time rather than loading all 6,557 images into memory — important given the dataset is fetched from URLs rather than stored locally.
- **NFStream integration:** Captures live network flows (TCP/UDP statistics, packet timing, byte counts) in a background thread while images are being downloaded. These flow features are merged into each record and could serve as an additional feature modality.
- **`preprocess_image` placeholder:** The function currently returns raw bytes unchanged. This is the hook where image resizing, normalisation, and augmentation will be implemented (see Section 4 for the planned implementation).
- **IP-based flow matching:** Flows are matched to download records by destination IP, with hostnames pre-resolved before the download loop starts.

**Note on NFStream:** Running this script requires either Administrator privileges (Windows + Npcap) or root/`CAP_NET_RAW` (Linux). The `INTERFACE` constant (`"Wi-Fi"` by default) must match the active network adapter.

The overall record structure yielded per image:

```python
{
    "url": str,
    "label": str,           # "REAL" or "FAKE"
    "dst_ip": str,
    "http_status": int,
    "content_length_bytes": int,
    "response_time_s": float,
    "image_data": np.ndarray,   # after preprocess_image is implemented
    "flow_features": dict,      # NFStream flow statistics, or None
}
```

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

With ~4,500 training samples, JL targets around 200–400 dimensions for reasonable distortion tolerance (ε ≈ 0.5–0.9). PCA can be pushed further (50–128 components) while retaining most variance. Both will be evaluated.

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
| **Frequency-domain SVM** | Extract DCT/FFT power spectrum from images before feeding to SVM. StyleGAN3 leaves known spectral artifacts — this is a domain-informed approach. |
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

Vision Transformers have shown strong results on deepfake detection benchmarks because self-attention can capture long-range spatial inconsistencies that GANs often introduce — particularly in background/hair regions that StyleGAN3 can render with subtle artifacts. EfficientNet is the backup if training budget is limited. Both benefit from the ImageNet pretraining.

### 🥈 Expected Second Tier: CNN Embeddings + Soft-SVM or XGBoost

A pretrained CNN (ResNet/EfficientNet) used purely as a feature extractor, with a Soft-SVM or gradient boosted tree on top, is a well-established and reliable pattern. With only ~4,500 training images, the limited fine-tuning data is less of an issue here since the CNN backbone stays frozen. This combination should outperform end-to-end fine-tuning if training data is too small to move the pretrained weights usefully.

### 🥉 Expected Third Tier: Frequency-domain features + Soft-SVM

StyleGAN3 is known to produce characteristic high-frequency artifacts in the DCT/FFT domain — something real camera images do not exhibit. A Soft-SVM on a frequency power spectrum is therefore a strong domain-informed baseline that may punch well above its computational weight. This is a particularly interesting result to report because it would demonstrate that the deepfakes are detectable without any deep learning at all.

### Predicted Weakest Configurations

- **Hard-SVM on raw pixels:** The hard margin assumption will almost certainly be violated with high-dimensional noisy pixel features. Expected to fail or require extremely aggressive dimensionality reduction.
- **Linear Regression with threshold:** Fundamental mismatch between the model's output space and the binary task. Included for completeness and as a pedagogical comparison point, not for performance.
- **Custom CNN trained from scratch:** With 4,500 images this will underfit badly. Expected to be clearly dominated by pretrained alternatives.
- **PCA/JL + Hard-SVM on raw pixels:** Even with dimensionality reduction, the hard-margin assumption is fragile. JL may help slightly more than PCA here (distance preservation), but neither will rescue Hard-SVM on raw pixels.

### The Most Interesting Predicted Finding

The **JL embedding vs PCA** comparison on Soft-SVM is expected to produce very similar results — JL's distance-preservation guarantee makes it theoretically equivalent for SVMs, but PCA's data-awareness may give it a slight edge at aggressive compression ratios. If JL matches PCA at, say, 300 dims, this is a practically useful result: JL requires no fitting and can be applied to new data instantly, making it preferable in a pipeline where the feature distribution may shift.

---

## 9. Open Questions for the Report

- Does including surviving metadata features (`confidence_score`, `gender`, `age_group`, `image_quality`) alongside image features improve any model? Or does the image signal dominate entirely?
- Does data augmentation help the deep models given the small training set, or does it introduce too much variance?
- Can the frequency-domain approach replicate or approach the accuracy of deep models, suggesting the artifacts are fundamentally spectral in nature?
- Does the choice of PCA vs JL matter more for Hard-SVM (where margins are strict) than for Soft-SVM?

---

*Document generated after exploratory data analysis. All predictions are pre-experiment hypotheses and will be revisited in the final report.*