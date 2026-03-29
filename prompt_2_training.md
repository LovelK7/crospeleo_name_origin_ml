# Prompt 2 — Model Training & Evaluation

**Copy-paste this entire prompt into Claude in VS Code as-is.**

---

## Context

This is the second step of the `crospeleo-name-origin-ml` project at:
`C:\Users\Lovel.IZRK-LK-NB\Programming\crospeleo-name-origin-ml`

In Prompt 1 we created:
- `data/processed/dataset_clean.csv` — cleaned dataset (~5967 rows with non-null labels)
- `data/processed/dataset_features.csv` — same + engineered features
- `src/data_loader.py` and `src/feature_engineering.py`

The target variable is `Podrijetlo imena` with 6 classes (heavily imbalanced):
- smišljeno novo: ~44%
- smišljeno prema toponimu: ~28%
- preuzeto kao lokalni naziv: ~21%
- preuzeto iz literature: ~3%
- preuzeto sa karte: ~1.6%
- nepoznato podrijetlo: ~1.6%

## Task — Step 2: Model training, tuning, and evaluation

Create these new files:

```
src/
  train.py            # main training script
  evaluate.py         # evaluation & reporting
  predict.py          # inference interface (for later use)
models/
  (saved model artifacts go here)
reports/
  (evaluation reports go here)
```

### train.py

Implement the following training pipeline:

**1. Data split:**
- Load `dataset_features.csv`
- Stratified train/test split: 80/20, `random_state=42`
- Print class distribution in both train and test sets

**2. Feature preparation — two parallel feature sets:**

**Set A: Hand-crafted features only**
All the engineered numeric columns from feature_engineering.py:
`name_word_count`, `name_char_count`, `name_has_number`, `name_has_generic_prefix`, `name_has_preposition`, `name_uppercase_ratio`, `name_is_single_word`, `name_has_possessive_suffix`, `name_lokalitet_overlap`, `name_mjesto_overlap`, `name_equals_lokalitet_token`, `name_equals_mjesto_token`, `sinonimi_exists`, `sinonimi_count`, `vrsta_objekta_encoded`, `name_looks_descriptive`, `name_looks_humorous_or_creative`

**Set B: TF-IDF features**
- Use `combined_text` column
- TF-IDF with: `max_features=5000`, `ngram_range=(1,3)`, `analyzer='char_wb'` (character n-grams capture Croatian morphology better than word n-grams)
- Also try a second TF-IDF with: `max_features=3000`, `ngram_range=(1,2)`, `analyzer='word'`, `min_df=2`

**Set C: Combined (Set A + Set B)**
- Concatenate hand-crafted features + both TF-IDF matrices using `scipy.sparse.hstack` or similar

**3. Models to train (all with `class_weight='balanced'` to handle imbalance):**

For each feature set (A, B, C), train:

1. **Logistic Regression** — `C=1.0`, `max_iter=1000`, `class_weight='balanced'`, `multi_class='multinomial'`
2. **Linear SVM** — `LinearSVC` with `class_weight='balanced'`, wrapped in `CalibratedClassifierCV` for probability estimates
3. **Random Forest** — `n_estimators=300`, `class_weight='balanced'`, `max_depth=None`
4. **Gradient Boosting** — use `HistGradientBoostingClassifier` with sample weights computed from class frequencies (since it doesn't have `class_weight` param directly)

That's 4 models × 3 feature sets = 12 experiments.

**4. Cross-validation:**
- Use `StratifiedKFold(n_splits=5)` on the training set
- Report mean and std of: macro F1, weighted F1, accuracy
- Use `cross_val_predict` to get out-of-fold predictions for the best model

**5. Track results in a comparison table:**
```
| Feature Set | Model | CV Macro-F1 | CV Weighted-F1 | CV Accuracy |
```

### evaluate.py

For the **best model** (highest CV macro-F1):

1. **Train on full training set, evaluate on held-out test set**

2. **Classification report** — `sklearn.metrics.classification_report` with all 6 class names

3. **Confusion matrix** — save as both:
   - Text table in `reports/confusion_matrix.txt`
   - Heatmap PNG in `reports/confusion_matrix.png` (use matplotlib/seaborn)

4. **Per-class analysis:**
   - Which classes are most confused with each other?
   - What is top-1 accuracy and top-2 accuracy per class?
   - Compute top-k accuracy: for each test sample, check if the true label is in the model's top-k predictions (k=1,2,3)

5. **Confidence analysis:**
   - Distribution of predicted probabilities for correct vs incorrect predictions
   - Suggest a confidence threshold below which the system should flag for human review
   - Save histogram of confidence scores to `reports/confidence_distribution.png`

6. **Error analysis sample:**
   - Print 20 misclassified examples showing: name, true label, predicted label, top-3 predictions with probabilities
   - Focus on errors in the minority classes

7. **Save full results report** to `reports/evaluation_report.txt`

### Important training notes

- **Handle imbalance carefully.** With only ~94-172 samples in the bottom 3 classes, macro-F1 is the primary metric (not accuracy). Accuracy will be misleadingly high because the model can score ~44% just by predicting "smišljeno novo" always.
- **Use `class_weight='balanced'`** on all models that support it.
- **For HistGradientBoosting**, compute sample weights manually: `sample_weight = n_samples / (n_classes * np.bincount(y_encoded))` mapped per sample.
- **Save the best model** using `joblib.dump()` to `models/best_model.joblib`
- **Save the fitted TF-IDF vectorizers** to `models/tfidf_char.joblib` and `models/tfidf_word.joblib`
- **Save the label encoder** for `Vrsta objekta` to `models/vrsta_encoder.joblib`
- **Save the feature engineering config** (column names, parameters) to `models/feature_config.json`
- **Save training lookup data** for nearest-neighbor explainer:
  - `models/train_tfidf_matrix.npz` — sparse TF-IDF char matrix of training set (use `scipy.sparse.save_npz`)
  - `models/train_lookup.csv` — with columns: `Ime objekta`, `Podrijetlo imena` (matching row order of the TF-IDF matrix)
- All text handling must preserve Croatian characters (č, ć, š, ž, đ)
- Print clear progress messages during training — this will take a few minutes

### Additional requirements in requirements.txt

```
matplotlib>=3.7
seaborn>=0.12
scipy>=1.10
joblib>=1.3
```

### Git

Commit after this step with message: "Add model training pipeline and evaluation"