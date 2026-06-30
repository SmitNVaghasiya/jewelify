# ML Pipeline Analysis & Improvement Plan

> Analysis date: 2026-06-30  
> Covers: `Jewelify_FNN_and_RL_codes.ipynb`, `Jewelify_Code.ipynb`, `Data/csv/`, datasets, libraries, and architecture recommendations.

---

## 1. Data — Critical Problems

**279 total samples.** That is tiny for a deep learning regression task.

**Class imbalance is severe:**

| Category | Count | % |
|---|---|---|
| Highly Compatible | 244 | 87.4% |
| Very Compatible | 15 | 5.4% |
| Slightly Compatible | 8 | 2.9% |
| Compatible | 6 | 2.1% |
| Not Compatible | 4 | 1.4% |
| Neutral | 2 | 0.7% |

Model learns "always predict high score" and still gets 87% accuracy. It cannot learn what *bad* looks like with only 4 negative examples. Mean rating is 0.94/1.0 — basically everything is labeled "perfect." **This is the biggest problem before anything else.**

---

## 2. The Cross-Pair Problem (Confirmed)

Current training: `combined_sorted_jpg` = pre-composed images (face + necklace + earring stitched into one photo). 279 specific combos were labeled.

**What is missing:** Training never covers all possible cross-combinations. With 150 faces, 150 necklaces, 150 earrings — that is 3.375M possible combos. Only 279 were labeled. The model has no concept of *why* necklace_A works with earring_B but not earring_C, because it sees each combination as an opaque blob.

The commented-out code in `Jewelify_Code.ipynb` Cell 8 shows this was known — the `itertools.product` approach was drafted but never implemented for training, only for inference. **This is the core architectural flaw.**

---

## 3. Feature Architecture Problems

### Combined image features = entangled features

Extracting MobileNetV2 features from a *combined* image (face+jewelry) means the feature vector mixes face structure with necklace style with earring color. They cannot be decomposed later.

**Correct approach:**
- Extract face features separately → 1280-dim vector
- Extract necklace features separately → 1280-dim vector
- Extract earring features separately → 1280-dim vector
- Concatenate → 3840-dim → train compatibility model

This way, at inference time: take 1 new face + 1 new jewelry → extract each independently → predict. Currently the model only works if the exact pre-computed combination exists.

### Preprocessing inconsistency

Old cells use `img / 255.0` (scales to [0,1]). Newer cells use MobileNetV2's `preprocess_input` (scales to [-1,1]). These give **incompatible feature spaces**. If stored `.npy` features were computed with one method and prediction uses the other, scores will be wrong.

**Fix:** Pick one (`preprocess_input`) and use it everywhere, including when recomputing all saved `.npy` files.

---

## 4. Model Issues

### Two-model ensemble is logically broken

`Jewelify_Code.ipynb` trains:
- Model A: necklace+earring compatibility only (no face)
- Model B: face+necklace+earring from 200 combined images
- Final score = average of both

Model A ignores face shape entirely. Model B trains on only 200 samples. Averaging them does not fix either problem.

### XGBoost + MLP ensemble (in production)

More principled architecture, but same data problems apply. Garbage in, garbage out.

---

## 5. Recommended File Structure

**File 1: `prepare_data.py`** — generate training features
```
- Load face images (separate folder)
- Load necklace images (separate folder)
- Load earring images (separate folder)
- Extract features for each independently (consistent preprocess_input)
- Save: face_features.npy, necklace_features.npy, earring_features.npy
- Generate all cross-pairs from labeled data
- Output: X_train.npy, y_train.npy
```

**File 2: `train_model.py`** — train from those features
```
- Load X_train.npy, y_train.npy
- Train XGBoost (fast, interpretable, good on tabular features)
- Train MLP (complementary)
- Ensemble predictions
- Save scaler + models
- Report: per-class accuracy, confusion matrix, R² score
```

Currently the notebooks mix data prep, training, and inference across 45 cells with lots of commented-out dead code. Separate scripts are essential.

---

## 6. Dataset Analysis

### Kaggle Datasets — Ranked by Usefulness

| Dataset | Rating | Reason |
|---|---|---|
| `jewelry-classification/data` (competition) | ⭐⭐⭐⭐⭐ | Competition-grade, clean images, already categorized — best starting point |
| `sapnilpatel/tanishq-jewellery-dataset` | ⭐⭐⭐⭐ | Tanishq = Indian jewelry brand = exact target demographic, high quality product photos |
| `sidd108/jewelry-detection-dataset` | ⭐⭐⭐ | Good earrings, weak necklaces — use earring subset only |
| `mylikes/jewelry-dataset` | ⭐⭐⭐ | General, useful for diversity |
| Fashion dataset (jewelry extraction) | ⭐⭐⭐⭐ | Jewelry worn in context = real-world styling signal, extract jewelry subimages |
| `harshjangid0015/jewelry-databaset` | ⭐⭐ | Likely scraped/noisy, inspect before using |
| `iammaidul/ringfir-dataset` | ⭐ | Rings only — not the use case (earrings + necklace), skip |

### Why Tanishq Dataset is an Underrated Pick

Indian bridal and traditional jewelry has specific style rules (Kundan pairs with Polki, heavy sets require matching earrings, etc.) that generic Western fashion datasets miss entirely. If users are buying Indian jewelry, this data matches the domain. Prioritize it.

### Other Sources Worth Considering

- **Polyvore dataset** — ~20K outfits with jewelry compatibility labels, publicly available
- **DeepFashion** — category-level annotations, large scale
- Pinterest "outfit of the day" scraping — jewelry worn together = natural positive pairs
- Instagram fashion hashtags — same signal

---

## 7. Modern Libraries — What Changed

**MobileNetV2 in 2025 is like using jQuery today. Not wrong, just obsolete.**

### Feature Extraction — Replace MobileNetV2

#### CLIP (OpenAI) — Biggest Upgrade Available

Already installed via `transformers 4.57.0`. Works now.

```python
from transformers import CLIPModel, CLIPProcessor

# Generic CLIP
model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14")

# Fashion-specific (recommended for Jewelify)
model = CLIPModel.from_pretrained("patrickjohncyh/fashion-clip")
```

- MobileNetV2 → 1280-dim generic image features
- CLIP → 768-dim features that understand "gold necklace on oval face" **semantically**
- FashionCLIP specifically trained on 700K+ fashion product images
- Zero-shot: no labels needed to get useful similarity scores

#### DINOv2 (Meta) — Best Pure Visual Features

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("facebook/dinov2-large")
```

- Self-supervised ViT trained on 142M images
- Better spatial understanding than MobileNetV2 — matters for jewelry shape and proportion
- No labels needed for feature extraction

#### timm (PyTorch Image Models)

```bash
pip install timm
```

```python
import timm
model = timm.create_model('convnext_large.fb_in22k', pretrained=True, num_classes=0)
# 500+ pretrained models, all better than MobileNetV2
```

### Face Analysis — Replace Haar Cascade

Current: Haar Cascade (2001 algorithm) → gives only "face detected yes/no"

Replace with **InsightFace**:
```bash
pip install insightface
```

InsightFace gives:
- Face shape: oval / round / square / heart
- Skin undertone: warm / cool
- Age estimate
- Face width:height ratio
- Landmark positions (jawline, forehead width, cheekbone width)

All of these are **actual jewelry recommendation signals**. A wide forehead = drop earrings recommended. Round face = long necklaces recommended. Current system ignores all of this.

### Already Installed — Use These

`pytorch-metric-learning 2.9.0` is already installed. This enables contrastive learning without any new dependencies:

```python
from pytorch_metric_learning import losses
loss = losses.NTXentLoss()
```

---

## 8. Do You Need CSV Labels? Honest Answer

**Your current CSV is actively hurting more than helping.**

Why:
- 87% "Highly Compatible" = biased ground truth
- Only 2 raters = not reliable
- 279 samples = model memorizes, does not generalize

### Three Paths — Ordered by Label Dependency

#### Path A: Zero-Shot CLIP (0 labels needed)

```python
face_embedding = CLIP(face_image)
jewelry_embedding = FashionCLIP(jewelry_image)
score = cosine_similarity(face_embedding, jewelry_embedding)
```

No training. No CSV. Works immediately. Not customized to your domain but better than a model trained on 87%-biased data.

**Realistic accuracy ceiling: ~65-70% on genuine compatibility judgments.** Good enough to ship an MVP, clear enough ceiling to motivate improvement.

#### Path B: Contrastive Learning (few labels needed)

`Data/src/Old/Wearing/` contains images of jewelry worn on faces. Those are **natural positive pairs** — if someone wore it together, it fits.

```python
# pytorch-metric-learning already installed
from pytorch_metric_learning import losses
loss = losses.NTXentLoss()
# Train: "face+jewelry combos photographed together = compatible"
# Random mismatches = negative pairs
```

Need ~50-100 "worn together" images as positives, randomly mismatched pairs as negatives. **No CSV rating needed.**

#### Path C: Fine-Tuned CLIP + Small Balanced Label Set (50-200 labels)

1. Start with FashionCLIP embeddings (already good)
2. Collect 200 *balanced* labels — 50 each across: not compatible / slightly / very / highly
3. Fine-tune a small head on top

**This beats the current 279-sample heavily-biased CSV immediately.** Balance matters more than quantity at this scale.

---

## 9. Priority Order — What to Do First

| Priority | Action | Why |
|---|---|---|
| 1 | Fix class imbalance | 87% bias makes everything else pointless |
| 2 | Download Tanishq + jewelry-classification datasets | Quality images, right category |
| 3 | Replace MobileNetV2 with FashionCLIP | Single library swap, immediate quality jump, `transformers` already installed |
| 4 | Replace Haar Cascade with InsightFace | Get face shape/skin tone as explicit features |
| 5 | Discard current CSV | Collect 200 balanced samples, 50 per class |
| 6 | Use contrastive loss on worn-jewelry images | `pytorch-metric-learning` already installed, free signal being ignored |
| 7 | Restructure into `prepare_data.py` + `train_model.py` | Clean separation of concerns |

**The library upgrade alone (MobileNetV2 → FashionCLIP) will improve feature quality more than doubling the dataset size would. Do that first.**

---

## 10. Architecture — What the New Pipeline Should Look Like

```
[Face Image]        [Necklace Image]    [Earring Image]
      |                    |                   |
 InsightFace          FashionCLIP         FashionCLIP
 (face shape,        (style, color,      (style, color,
  skin tone,          material           material
  landmarks)          embeddings)         embeddings)
      |                    |                   |
   face_vec             neck_vec            ear_vec
   (explicit +          (768-dim)           (768-dim)
    embedding)
      |                    |                   |
      └────────────────────┴───────────────────┘
                           |
                    Concatenate → ~2000-dim
                           |
                  XGBoost + MLP ensemble
                           |
                  Compatibility score 0-1
                           |
                  Category + Recommendations
```

---

## 11. Notes on Current Production Code

- `Jewelify_server/services/predictor.py` uses XGBoost + Keras MLP ensemble — architecture is correct, data is the problem
- `predictor.py` calls `asyncio.to_thread()` correctly for ML inference — keep this
- Feature files (`face_features.npy`, etc.) were deleted from git (per gitignore) — need to regenerate with new pipeline
- `scaler_mlp_v1.pkl` and `scaler_xgboost_v1.pkl` — if feature extraction method changes (MobileNetV2 → CLIP), scalers must be retrained from scratch, old scalers will give wrong results
