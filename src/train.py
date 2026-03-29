"""
Training pipeline: 4 models × 3 feature sets = 12 experiments with 5-fold CV.
Selects the best model by CV macro-F1 and saves all artifacts.
"""

import json
import sys
import time
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
MODELS_DIR.mkdir(exist_ok=True)

TARGET_COL = "Podrijetlo imena"
TEXT_COL = "combined_text"
RANDOM_STATE = 42

HANDCRAFTED_FEATURES = [
    "name_word_count", "name_char_count", "name_has_number",
    "name_has_generic_prefix", "name_has_preposition", "name_uppercase_ratio",
    "name_is_single_word", "name_has_possessive_suffix", "name_lokalitet_overlap",
    "name_mjesto_overlap", "name_equals_lokalitet_token", "name_equals_mjesto_token",
    "sinonimi_exists", "sinonimi_count", "vrsta_objekta_encoded",
    "name_looks_descriptive", "name_looks_humorous_or_creative",
]


# ---------------------------------------------------------------------------
# Data loading & splitting
# ---------------------------------------------------------------------------

def load_and_split(features_csv: Path):
    print(f"Loading {features_csv.name} ...")
    df = pd.read_csv(features_csv, encoding="utf-8-sig", dtype=str)

    for col in HANDCRAFTED_FEATURES:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

    le_target = LabelEncoder()
    y = le_target.fit_transform(df[TARGET_COL])

    df_train, df_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nTrain: {len(df_train):,}  |  Test: {len(df_test):,}")
    print("\nClass distribution — TRAIN:")
    for i, cls in enumerate(le_target.classes_):
        n = (y_train == i).sum()
        print(f"  {cls:<35} {n:>4}  ({n/len(y_train)*100:.1f}%)")
    print("\nClass distribution — TEST:")
    for i, cls in enumerate(le_target.classes_):
        n = (y_test == i).sum()
        print(f"  {cls:<35} {n:>4}  ({n/len(y_test)*100:.1f}%)")

    return df_train, df_test, y_train, y_test, le_target


# ---------------------------------------------------------------------------
# Feature sets
# ---------------------------------------------------------------------------

def build_feature_sets(df_train, df_test):
    """Fit TF-IDF on training set; build A/B/C matrices for train and test."""

    X_train_A = df_train[HANDCRAFTED_FEATURES].values.astype(float)
    X_test_A  = df_test[HANDCRAFTED_FEATURES].values.astype(float)

    train_texts = df_train[TEXT_COL].fillna("").tolist()
    test_texts  = df_test[TEXT_COL].fillna("").tolist()

    print("\nFitting TF-IDF char n-grams (max_features=5000, ngram=(1,3)) ...")
    tfidf_char = TfidfVectorizer(max_features=5000, ngram_range=(1, 3), analyzer="char_wb")
    X_train_char = tfidf_char.fit_transform(train_texts)
    X_test_char  = tfidf_char.transform(test_texts)

    print("Fitting TF-IDF word n-grams (max_features=3000, ngram=(1,2)) ...")
    tfidf_word = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), analyzer="word", min_df=2)
    X_train_word = tfidf_word.fit_transform(train_texts)
    X_test_word  = tfidf_word.transform(test_texts)

    # B = char + word TF-IDF
    X_train_B = sp.hstack([X_train_char, X_train_word], format="csr")
    X_test_B  = sp.hstack([X_test_char,  X_test_word],  format="csr")

    # C = handcrafted + B
    X_train_C = sp.hstack([sp.csr_matrix(X_train_A), X_train_char, X_train_word], format="csr")
    X_test_C  = sp.hstack([sp.csr_matrix(X_test_A),  X_test_char,  X_test_word],  format="csr")

    # Dense SVD reduction for HistGradientBoosting (doesn't support sparse in sklearn 1.8)
    print("Fitting TruncatedSVD(300) for HistGradientBoosting dense variants ...")
    svd = TruncatedSVD(n_components=300, random_state=RANDOM_STATE)
    X_train_B_dense = svd.fit_transform(X_train_B)
    X_test_B_dense  = svd.transform(X_test_B)
    X_train_C_dense = np.hstack([X_train_A, X_train_B_dense])
    X_test_C_dense  = np.hstack([X_test_A,  X_test_B_dense])

    print(f"\nFeature set sizes:")
    print(f"  A (hand-crafted):         {X_train_A.shape[1]:>5}")
    print(f"  B (TF-IDF sparse):        {X_train_B.shape[1]:>5}")
    print(f"  C (combined sparse):      {X_train_C.shape[1]:>5}")
    print(f"  B_dense (SVD-300):        {X_train_B_dense.shape[1]:>5}  [HistGB only]")
    print(f"  C_dense (A+SVD-300):      {X_train_C_dense.shape[1]:>5}  [HistGB only]")

    feature_sets = {
        "A_handcrafted": (X_train_A,       X_test_A),
        "B_tfidf":       (X_train_B,       X_test_B),
        "C_combined":    (X_train_C,       X_test_C),
    }
    # Dense alternatives used only by HistGB (key matches parent feature set name)
    dense_feature_sets = {
        "A_handcrafted": (X_train_A,       X_test_A),        # already dense
        "B_tfidf":       (X_train_B_dense, X_test_B_dense),
        "C_combined":    (X_train_C_dense, X_test_C_dense),
    }
    return feature_sets, dense_feature_sets, tfidf_char, tfidf_word, svd


# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

def get_models(y_train):
    sw = compute_sample_weight("balanced", y_train)
    return {
        "LogisticRegression": {
            "model": LogisticRegression(
                C=1.0, max_iter=1000, class_weight="balanced",
                solver="lbfgs", random_state=RANDOM_STATE,
            ),
            "fit_params": {},
        },
        "LinearSVM": {
            "model": CalibratedClassifierCV(
                LinearSVC(class_weight="balanced", max_iter=2000, random_state=RANDOM_STATE),
                cv=3,
            ),
            "fit_params": {},
        },
        "RandomForest": {
            "model": RandomForestClassifier(
                n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=RANDOM_STATE,
            ),
            "fit_params": {},
        },
        "HistGradientBoosting": {
            "model": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            "fit_params": {"sample_weight": sw},
        },
    }


# ---------------------------------------------------------------------------
# Cross-validation
# ---------------------------------------------------------------------------

def run_cv(model, X_train, y_train, use_sample_weight=False):
    from sklearn.metrics import f1_score, accuracy_score
    import sklearn.base as skbase

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    if not use_sample_weight:
        scoring = {"macro_f1": "f1_macro", "weighted_f1": "f1_weighted", "accuracy": "accuracy"}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_validate(model, X_train, y_train, cv=skf, scoring=scoring, n_jobs=1)
        return {
            "macro_f1_mean":    scores["test_macro_f1"].mean(),
            "macro_f1_std":     scores["test_macro_f1"].std(),
            "weighted_f1_mean": scores["test_weighted_f1"].mean(),
            "weighted_f1_std":  scores["test_weighted_f1"].std(),
            "accuracy_mean":    scores["test_accuracy"].mean(),
            "accuracy_std":     scores["test_accuracy"].std(),
        }

    # Manual CV for HistGradientBoosting (needs sample_weight per fold, dense input)
    macro_f1s, weighted_f1s, accs = [], [], []
    X_dense = X_train.toarray() if sp.issparse(X_train) else X_train
    for train_idx, val_idx in skf.split(X_dense, y_train):
        X_tr, X_val = X_dense[train_idx], X_dense[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        sw_fold = compute_sample_weight("balanced", y_tr)
        clone = skbase.clone(model)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            clone.fit(X_tr, y_tr, sample_weight=sw_fold)
        y_pred = clone.predict(X_val)
        macro_f1s.append(f1_score(y_val, y_pred, average="macro", zero_division=0))
        weighted_f1s.append(f1_score(y_val, y_pred, average="weighted", zero_division=0))
        accs.append(accuracy_score(y_val, y_pred))

    return {
        "macro_f1_mean":    np.mean(macro_f1s),
        "macro_f1_std":     np.std(macro_f1s),
        "weighted_f1_mean": np.mean(weighted_f1s),
        "weighted_f1_std":  np.std(weighted_f1s),
        "accuracy_mean":    np.mean(accs),
        "accuracy_std":     np.std(accs),
    }


def run_all_experiments(feature_sets, dense_feature_sets, y_train):
    """
    12 experiments: 4 models × 3 feature sets.
    HistGradientBoosting uses SVD-reduced dense variants of B and C
    (sklearn 1.8 dropped sparse support for HistGB).
    """
    models_def = get_models(y_train)
    results = []
    total = len(feature_sets) * len(models_def)
    i = 0
    for fs_name in feature_sets:
        for model_name, model_def in models_def.items():
            i += 1
            is_hgb = model_name == "HistGradientBoosting"
            # HistGB uses dense (SVD) feature sets; all others use sparse
            X_train, _ = dense_feature_sets[fs_name] if is_hgb else feature_sets[fs_name]
            fs_label = f"{fs_name}{'[dense]' if is_hgb and fs_name != 'A_handcrafted' else ''}"
            print(f"\n[{i}/{total}] {model_name} | {fs_label} ...")
            t0 = time.time()
            scores = run_cv(
                model_def["model"], X_train, y_train,
                use_sample_weight=is_hgb,
            )
            elapsed = time.time() - t0
            print(
                f"  Macro-F1: {scores['macro_f1_mean']:.4f} ± {scores['macro_f1_std']:.4f}"
                f"  | Weighted-F1: {scores['weighted_f1_mean']:.4f}"
                f"  | Acc: {scores['accuracy_mean']:.4f}"
                f"  | {elapsed:.0f}s"
            )
            results.append({"Feature Set": fs_name, "Model": model_name, **scores, "time_s": round(elapsed, 1)})
    return results


def print_results_table(results):
    print("\n" + "=" * 105)
    print("RESULTS TABLE (sorted by CV Macro-F1)")
    print("=" * 105)
    print(f"{'Feature Set':<18} {'Model':<25} {'CV Macro-F1':>18} {'CV Weighted-F1':>20} {'CV Accuracy':>16}")
    print("-" * 105)
    for r in sorted(results, key=lambda x: x["macro_f1_mean"], reverse=True):
        print(
            f"{r['Feature Set']:<18} {r['Model']:<25}"
            f"  {r['macro_f1_mean']:.4f} ± {r['macro_f1_std']:.3f}"
            f"      {r['weighted_f1_mean']:.4f} ± {r['weighted_f1_std']:.3f}"
            f"    {r['accuracy_mean']:.4f} ± {r['accuracy_std']:.3f}"
        )
    print("=" * 105)


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------

def save_artifacts(
    best_model_name, best_fs_name, feature_sets, dense_feature_sets, y_train,
    tfidf_char, tfidf_word, svd, le_target, df_train, results, models_def,
):
    print(f"\nFitting best model ({best_model_name} / {best_fs_name}) on full training set ...")
    is_hgb = best_model_name == "HistGradientBoosting"
    X_train_best, _ = dense_feature_sets[best_fs_name] if is_hgb else feature_sets[best_fs_name]

    best_model = models_def[best_model_name]["model"]
    if is_hgb:
        sw = compute_sample_weight("balanced", y_train)
        best_model.fit(X_train_best, y_train, sample_weight=sw)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            best_model.fit(X_train_best, y_train)

    joblib.dump(best_model, MODELS_DIR / "best_model.joblib")
    print(f"  Saved best_model.joblib")

    joblib.dump(tfidf_char, MODELS_DIR / "tfidf_char.joblib")
    joblib.dump(tfidf_word, MODELS_DIR / "tfidf_word.joblib")
    print(f"  Saved tfidf_char.joblib, tfidf_word.joblib")

    joblib.dump(le_target, MODELS_DIR / "label_encoder.joblib")
    print(f"  Saved label_encoder.joblib")

    # Vrsta objekta encoder — fit on train split only
    vrsta_le = LabelEncoder()
    vrsta_le.fit(df_train["Vrsta objekta"].fillna("unknown").tolist())
    joblib.dump(vrsta_le, MODELS_DIR / "vrsta_encoder.joblib")
    print(f"  Saved vrsta_encoder.joblib")

    if svd is not None:
        joblib.dump(svd, MODELS_DIR / "svd.joblib")
        print(f"  Saved svd.joblib")

    config = {
        "handcrafted_features": HANDCRAFTED_FEATURES,
        "tfidf_char_params":  {"max_features": 5000, "ngram_range": [1, 3], "analyzer": "char_wb"},
        "tfidf_word_params":  {"max_features": 3000, "ngram_range": [1, 2], "analyzer": "word", "min_df": 2},
        "target_column":      TARGET_COL,
        "text_column":        TEXT_COL,
        "best_model":         best_model_name,
        "best_feature_set":   best_fs_name,
        "uses_svd":           is_hgb and best_fs_name != "A_handcrafted",
        "classes":            list(le_target.classes_),
    }
    with open(MODELS_DIR / "feature_config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)
    print(f"  Saved feature_config.json")

    # Training TF-IDF char matrix for nearest-neighbour explainer
    X_train_char = tfidf_char.transform(df_train[TEXT_COL].fillna("").tolist())
    sp.save_npz(MODELS_DIR / "train_tfidf_matrix.npz", X_train_char)

    lookup = df_train[["Ime objekta", TARGET_COL]].reset_index(drop=True)
    lookup.to_csv(MODELS_DIR / "train_lookup.csv", index=False, encoding="utf-8-sig")
    print(f"  Saved train_tfidf_matrix.npz, train_lookup.csv")

    pd.DataFrame(results).to_csv(MODELS_DIR / "training_results.csv", index=False, encoding="utf-8-sig")
    print(f"  Saved training_results.csv")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    df_train, df_test, y_train, y_test, le_target = load_and_split(
        DATA_DIR / "dataset_features.csv"
    )

    feature_sets, dense_feature_sets, tfidf_char, tfidf_word, svd = build_feature_sets(df_train, df_test)

    print("\n" + "=" * 60)
    print("Running 12 experiments (4 models × 3 feature sets) — 5-fold CV")
    print("Note: HistGB uses SVD(300) dense reduction for B/C feature sets")
    print("=" * 60)
    results = run_all_experiments(feature_sets, dense_feature_sets, y_train)

    print_results_table(results)

    best = max(results, key=lambda r: r["macro_f1_mean"])
    print(f"\nBest: {best['Model']} on {best['Feature Set']} — CV Macro-F1 = {best['macro_f1_mean']:.4f}")

    models_def = get_models(y_train)
    save_artifacts(
        best["Model"], best["Feature Set"], feature_sets, dense_feature_sets, y_train,
        tfidf_char, tfidf_word, svd, le_target, df_train, results, models_def,
    )

    # Save test set for evaluate.py
    df_test_out = df_test.copy()
    df_test_out["_y_true"] = y_test
    df_test_out.to_csv(DATA_DIR / "test_set.csv", index=False, encoding="utf-8-sig")
    print(f"Saved test set: data/processed/test_set.csv")

    print("\nTraining complete.")
    return results, best


if __name__ == "__main__":
    run()
