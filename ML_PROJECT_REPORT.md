# ML Project Report — CroSpeleo Name Origin Classifier

**Date:** 2026-03-29
**Project:** `crospeleo_name_origin_ml`
**Task:** Multi-class classification of the `Podrijetlo imena` (name origin) field for Croatian speleological objects from the CroSpeleo cadastre.

---

## 1. Problem Definition

**Target field:** `Podrijetlo imena` (origin of name)
**Type:** Multi-class classification, 6 classes
**Source data:** `CroSpeleo - objekti.xlsx`, sheet `Objekti`, ~6370 rows

| Class | Train count | % |
|---|---|---|
| smišljeno novo | 2105 | 44.1% |
| smišljeno prema toponimu | 1362 | 28.5% |
| preuzeto kao lokalni naziv | 1015 | 21.3% |
| preuzeto iz literature | 138 | 2.9% |
| preuzeto sa karte | 78 | 1.6% |
| nepoznato podrijetlo | 75 | 1.6% |

**Key challenge:** Heavily imbalanced — the bottom 3 classes together are ~6% of the data. Primary metric is **macro-F1** (not accuracy) to penalise poor minority-class performance.

**Input features used:**
- `Ime objekta` (cave name) — primary signal
- `Sinonimi`, `Lokalitet`, `Najbliže mjesto`, `Vrsta objekta`, `Napomena (osnovni podaci)`

---

## 2. What Was Built

### Step 1 — Data Preparation & Feature Engineering (`src/data_loader.py`, `src/feature_engineering.py`)

- Loaded and cleaned the Excel source; 403 rows dropped for missing/invalid target → **5,967 usable rows**
- Engineered **17 hand-crafted features** covering:
  - Name structure (word count, char count, digit presence, uppercase ratio, single-word flag)
  - Croatian morphology (generic speleological prefixes, spatial prepositions, possessive suffixes)
  - Cross-reference overlap (Jaccard similarity and token match against `Lokalitet` and `Najbliže mjesto`)
  - Synonym and object-type features
- Built `combined_text` column (name + synonyms + location fields) for TF-IDF
- Outputs: `data/processed/dataset_clean.csv`, `data/processed/dataset_features.csv`

### Step 2 — Model Training & Evaluation (`src/train.py`, `src/evaluate.py`)

- **80/20 stratified split** (random_state=42): 4,773 train / 1,194 test
- **3 feature sets:**
  - A: 17 hand-crafted features (dense)
  - B: char n-gram TF-IDF (max 5000) + word TF-IDF (max 3000) → 8,000 sparse features
  - C: A + B combined (8,017 features)
- **4 models**, all with `class_weight='balanced'`:
  - Logistic Regression (lbfgs, C=1.0)
  - Linear SVM (CalibratedClassifierCV for probability estimates)
  - Random Forest (300 trees)
  - HistGradientBoosting (sample weights; uses TruncatedSVD(300) dense reduction for TF-IDF sets due to sklearn 1.8 sparse incompatibility)
- **5-fold stratified CV** on training set

#### 12-Experiment CV Results (sorted by macro-F1)

| Feature Set | Model | CV Macro-F1 | CV Weighted-F1 | CV Accuracy |
|---|---|---|---|---|
| **C_combined** | **LogisticRegression** | **0.4150 ± 0.006** | **0.6771** | **0.6681** |
| B_tfidf | LogisticRegression | 0.3968 ± 0.017 | 0.6370 | 0.6250 |
| C_combined | HistGradientBoosting | 0.3632 ± 0.015 | 0.6941 | 0.7163 |
| C_combined | RandomForest | 0.3571 ± 0.013 | 0.6646 | 0.6937 |
| C_combined | LinearSVM | 0.3494 ± 0.011 | 0.6768 | 0.7012 |
| ... | ... | ... | ... | ... |
| A_handcrafted | LogisticRegression | 0.2765 ± 0.009 | 0.5107 | 0.4701 |

**Winner: Logistic Regression on feature set C (hand-crafted + TF-IDF)**

### Step 3 — Prediction Interface & Integration Export (`src/predict.py`, `src/integration.py`, `src/validate_model.py`)

- `OriginPredictor` class with full prediction dict output
- Croatian natural-language explanations (rule-based, based on feature signals)
- Nearest-neighbour explainer: top-5 most similar training examples by cosine similarity on char TF-IDF
- `needs_user_confirmation` flag (threshold 0.6; hard classes flagged unless > 0.8)
- CLI: single prediction and batch CSV mode
- `format_for_dossier()` — wraps output in the `crospeleo-automation` dossier schema
- Validated on 10 hardcoded test cases

---

## 3. Final Model Performance (Held-Out Test Set, n=1,194)

| Metric | Value |
|---|---|
| **Accuracy** | 67.5% |
| **Macro F1** | 0.470 |
| **Weighted F1** | 0.687 |
| **Top-2 accuracy** | 84.4% |
| **Top-3 accuracy** | 92.6% |

### Per-class breakdown

| Class | Precision | Recall | F1 | n (test) |
|---|---|---|---|---|
| smišljeno novo | 0.840 | 0.736 | 0.785 | 527 |
| smišljeno prema toponimu | 0.707 | 0.692 | 0.699 | 341 |
| preuzeto kao lokalni naziv | 0.567 | 0.618 | 0.591 | 254 |
| preuzeto iz literature | 0.219 | 0.412 | 0.286 | 34 |
| preuzeto sa karte | 0.219 | 0.368 | 0.275 | 19 |
| nepoznato podrijetlo | 0.160 | 0.211 | 0.182 | 19 |

### Key observations

- The two dominant classes (smišljeno novo, smišljeno prema toponimu) are well-classified (F1 > 0.70)
- The 3 minority classes (preuzeto iz literature, preuzeto sa karte, nepoznato podrijetlo) have poor F1 (0.18–0.29) due to small training samples (~75–138 each) and inherent label ambiguity — these are virtually indistinguishable from surface features alone
- **Top-3 accuracy is 92.6%** — the true class is almost always in the model's top-3, which makes the interface useful even where top-1 is wrong
- **Confidence threshold 0.65:** predictions above this are correct ~86.5% of the time

### Most confused class pairs

| True | Predicted | Count |
|---|---|---|
| smišljeno novo | preuzeto kao lokalni naziv | 62 |
| smišljeno novo | smišljeno prema toponimu | 50 |
| smišljeno prema toponimu | preuzeto kao lokalni naziv | 42 |

---

## 4. Model Storage Format

All artifacts are saved with **joblib** (standard sklearn serialisation). The `models/` directory is the complete, self-contained model package.

| File | Size | Purpose |
|---|---|---|
| `models/best_model.joblib` | 377 KB | Fitted LogisticRegression |
| `models/tfidf_char.joblib` | 162 KB | Char n-gram TF-IDF vectorizer (fitted on train set) |
| `models/tfidf_word.joblib` | 117 KB | Word n-gram TF-IDF vectorizer (fitted on train set) |
| `models/svd.joblib` | 18.3 MB | TruncatedSVD(300) — saved but NOT used by best model |
| `models/label_encoder.joblib` | 1 KB | Target class label encoder |
| `models/vrsta_encoder.joblib` | 1 KB | `Vrsta objekta` categorical encoder |
| `models/feature_config.json` | 1 KB | Feature names, params, best_feature_set, class list |
| `models/train_tfidf_matrix.npz` | 3.7 MB | Sparse char TF-IDF of training set (for NN explainer) |
| `models/train_lookup.csv` | 180 KB | Training names + labels (for NN explainer) |

> **Note:** `svd.joblib` (18 MB) is large but only used if the best model were HistGradientBoosting. Since the winner is Logistic Regression, the SVD is loaded but not applied during inference. It can be omitted from deployment if size is a concern (set `uses_svd: false` in `feature_config.json`).

**Total model package size (excluding svd.joblib):** ~4.5 MB

---

## 5. How to Use the Predictor

### Python API

```python
from src.predict import OriginPredictor

predictor = OriginPredictor(model_dir="models/")

result = predictor.predict(
    ime_objekta    = "Jama pod Vršićem",
    lokalitet      = "Vršić",
    najblize_mjesto= "",
    sinonimi       = "",
    vrsta_objekta  = "",
    napomena       = "",
)
```

**Return format:**
```python
{
    "predicted_value":        "smišljeno prema toponimu",
    "confidence":             0.643,
    "top_k": [
        {"value": "smišljeno prema toponimu", "score": 0.643},
        {"value": "preuzeto kao lokalni naziv", "score": 0.226},
        ...  # all 6 classes
    ],
    "explanation": [
        "naziv sadrži prijedlog 'pod' — upućuje na toponimsko podrijetlo"
    ],
    "similar_examples": [
        {"name": "Jama na Vršiću", "origin": "smišljeno prema toponimu", "similarity": 0.497},
        ...  # top 5
    ],
    "needs_user_confirmation": False
}
```

### Dossier integration

```python
from src.integration import format_for_dossier
dossier_field = format_for_dossier(result)
# Returns: {"key": "podrijetlo_imena", "label": "Podrijetlo imena", "required": True, ...}
```

### CLI (single)

```bash
python src/predict.py --name "Jama pod Vršićem" --lokalitet "Vršić" --mjesto ""
```

### CLI (batch)

```bash
python src/predict.py --batch data/processed/new_objects.csv --output predictions.csv
```

Input CSV columns: `Ime objekta`, `Sinonimi`, `Lokalitet`, `Najbliže mjesto`, `Vrsta objekta`, `Napomena (osnovni podaci)`
Output adds: `predicted_origin`, `confidence`, `top_2`, `top_3`, `needs_confirmation`

---

## 6. What Must Be Exported for External Deployment

To use the predictor in another project or environment, copy the entire `models/` directory and the `src/predict.py` + `src/integration.py` files (plus `src/utils.py` for the tokenizer).

**Minimum required files:**

```
models/
  best_model.joblib        # the classifier
  tfidf_char.joblib        # char TF-IDF vectorizer
  tfidf_word.joblib        # word TF-IDF vectorizer
  label_encoder.joblib     # target class labels
  vrsta_encoder.joblib     # Vrsta objekta encoder
  feature_config.json      # feature/class configuration
  train_tfidf_matrix.npz   # for nearest-neighbour explainer
  train_lookup.csv         # for nearest-neighbour explainer

src/
  predict.py               # OriginPredictor class + CLI
  integration.py           # format_for_dossier()
  utils.py                 # tokenizer helpers
```

**Python dependencies (pip install):**
```
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
scipy>=1.10
joblib>=1.3
```

**No GPU required. No internet required at inference time. Inference is fast (milliseconds per sample).**

---

## 7. Limitations & Recommendations

1. **Minority classes are weak.** With only 75–138 training samples for `nepoznato podrijetlo`, `preuzeto sa karte`, and `preuzeto iz literature`, the model cannot reliably distinguish them. `needs_user_confirmation` is always `True` for these classes unless confidence > 0.80.

2. **Top-3 is the practical metric.** The orchestrator should present the top-3 predictions to the user rather than auto-filling — the correct answer is in the top-3 92.6% of the time.

3. **Active learning opportunity.** Every human correction is a valuable training sample. Re-training periodically with newly confirmed labels — especially for the 3 weak classes — will improve performance over time.

4. **The model is a character-level model.** It primarily relies on the name's character n-grams and morphology. It does not use semantic embeddings. Replacing TF-IDF with a Croatian language model (e.g., CroSloEngual-BERT) could improve especially the minority classes.
