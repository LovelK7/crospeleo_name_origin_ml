# Prompt 3 — Prediction Interface & Integration Export

**Copy-paste this entire prompt into Claude in VS Code as-is.**

---

## Context

This is the third and final step of the `crospeleo-name-origin-ml` project at:
crospeleo-name-origin-ml

In previous steps we:
1. Prepared data and engineered features (`data_loader.py`, `feature_engineering.py`)
2. Trained and evaluated multiple models, saved the best one to `models/best_model.joblib` along with vectorizers and encoders

The model predicts `Podrijetlo imena` (origin of name) for Croatian cave objects — one of 6 classes.

## Task — Step 3: Prediction interface, nearest-neighbor explainer, and integration-ready API

### 1. predict.py — Single-object prediction interface

Create `src/predict.py` with a class `OriginPredictor` that:

```python
class OriginPredictor:
    def __init__(self, model_dir: str = "models/"):
        """Load saved model, vectorizers, encoders, and training data for NN lookup."""
        ...

    def predict(self, ime_objekta: str,
                lokalitet: str = "",
                najblize_mjesto: str = "",
                sinonimi: str = "",
                vrsta_objekta: str = "",
                napomena: str = "") -> dict:
        """
        Returns prediction dict in this exact format:
        {
            "predicted_value": "smišljeno prema toponimu",
            "confidence": 0.78,
            "top_k": [
                {"value": "smišljeno prema toponimu", "score": 0.78},
                {"value": "preuzeto sa karte", "score": 0.14},
                {"value": "nepoznato podrijetlo", "score": 0.08}
            ],
            "explanation": [
                "ime dijeli glavni leksički korijen s lokalitetom",
                "struktura naziva odgovara konstruiranju prema toponimu"
            ],
            "similar_examples": [
                {"name": "Jama pod Obručem", "origin": "smišljeno prema toponimu", "similarity": 0.85},
                {"name": "Špilja pod Kukom", "origin": "smišljeno prema toponimu", "similarity": 0.82},
                ...
            ],
            "needs_user_confirmation": true
        }
        """
```

**Key design decisions:**

- `top_k`: Always return all 6 classes sorted by probability descending
- `confidence`: The probability of the top prediction
- `needs_user_confirmation`: `True` if confidence < 0.6 (configurable threshold), `True` always for the 3 hard-to-distinguish classes ("preuzeto kao lokalni naziv", "preuzeto sa karte", "preuzeto iz literature") unless confidence > 0.8
- `explanation`: Generate 1-3 short human-readable reasons in Croatian, based on which features fired. Examples:
  - "naziv sadrži prijedlog 'kod' — upućuje na toponimsko podrijetlo"
  - "naziv je jednočlani negeografski izraz — upućuje na smišljeno novo ime"
  - "naziv se poklapa s lokalitetom"
  - "sinonim postoji — moguće preuzeto iz literature"
  - "nema jasnih signala — preporuča se ručna provjera"
- `similar_examples`: Find the 5 most similar objects from the training set using cosine similarity on the TF-IDF char features. Return their name, actual origin label, and similarity score. This gives the user intuitive context.

### 2. Nearest-neighbor explainer

To support `similar_examples`, during `__init__`:
- Load the training data TF-IDF matrix (save it in Prompt 2 as `models/train_tfidf_matrix.npz` using `scipy.sparse.save_npz`)
- Load the corresponding training names and labels (save as `models/train_lookup.csv`)
- At prediction time, transform the new input with the same TF-IDF vectorizer, compute cosine similarity against all training vectors, return top-5

### 3. CLI interface

Add a simple CLI to `src/predict.py` so it can be tested from terminal:

```bash
python src/predict.py --name "Jama pod Vršićem" --lokalitet "Vršić" --mjesto "Kranjska Gora"
```

Output should be a pretty-printed JSON of the prediction result.

Also add a batch mode:

```bash
python src/predict.py --batch data/processed/new_objects.csv --output data/processed/predictions.csv
```

Where `new_objects.csv` has columns: `Ime objekta`, `Sinonimi`, `Lokalitet`, `Najbliže mjesto`, `Vrsta objekta`, `Napomena (osnovni podaci)`

And `predictions.csv` adds columns: `predicted_origin`, `confidence`, `top_2`, `top_3`, `needs_confirmation`

### 4. Integration JSON schema

Create `src/integration.py` that exports the prediction in the exact format expected by the `crospeleo-automation` project's dossier:

```python
def format_for_dossier(prediction: dict) -> dict:
    """
    Returns the dossier-compatible manual_choice_field format:
    {
        "key": "podrijetlo_imena",
        "label": "Podrijetlo imena",
        "required": true,
        "predicted_value": "smišljeno prema toponimu",
        "confidence": 0.78,
        "top_k": [...],
        "explanation": [...],
        "similar_examples": [...],
        "needs_user_confirmation": true,
        "source": "ml_model_v1"
    }
    """
```

### 5. Quick validation script

Create `src/validate_model.py` that:
1. Loads the saved model
2. Runs predictions on 10 hardcoded test cases (mix of easy and hard):

```python
test_cases = [
    {"name": "Jama pod Vršićem", "lokalitet": "Vršić", "mjesto": ""},
    {"name": "Acronium", "lokalitet": "", "mjesto": ""},
    {"name": "Špilja kod Marijanovića kuća", "lokalitet": "", "mjesto": ""},
    {"name": "Židovske jame", "lokalitet": "Rupe", "mjesto": "Pasanska Gorica"},
    {"name": "Plodni dan", "lokalitet": "Bijele Stijene", "mjesto": "Delnice"},
    {"name": "Maklenska", "lokalitet": "Maklenske njive", "mjesto": "Brod Moravice"},
    {"name": "Borušnjak 2", "lokalitet": "", "mjesto": ""},
    {"name": "Grbina peć", "lokalitet": "Lesina, Ćićarija", "mjesto": "Buzet"},
    {"name": "Konzerva", "lokalitet": "Sredenji Velebit", "mjesto": ""},
    {"name": "Spilja u Japagama", "lokalitet": "", "mjesto": "Krašić"},
]
```

Print each prediction nicely: name → predicted class (confidence) + top-3 + similar examples.

### 6. Save training lookup data (add to train.py if not already done)

Make sure Prompt 2's training script also saves:
- `models/train_tfidf_matrix.npz` — sparse TF-IDF matrix of training set (char n-gram vectorizer)
- `models/train_lookup.csv` — with columns: `Ime objekta`, `Podrijetlo imena` (matching rows of the TF-IDF matrix)

If these weren't saved in Prompt 2, add the saving logic to `train.py` now.

### Git

Commit with message: "Add prediction interface, NN explainer, and integration export"

### Final project structure should be:

```
crospeleo-name-origin-ml/
  README.md
  requirements.txt
  .gitignore
  data/
    raw/
    processed/
      dataset_clean.csv
      dataset_features.csv
  models/
    best_model.joblib
    tfidf_char.joblib
    tfidf_word.joblib
    vrsta_encoder.joblib
    feature_config.json
    train_tfidf_matrix.npz
    train_lookup.csv
  reports/
    evaluation_report.txt
    confusion_matrix.txt
    confusion_matrix.png
    confidence_distribution.png
  src/
    __init__.py
    data_loader.py
    feature_engineering.py
    train.py
    evaluate.py
    predict.py
    integration.py
    validate_model.py
    utils.py
```