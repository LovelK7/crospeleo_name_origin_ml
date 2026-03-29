"""
Evaluation and reporting for the best trained model.
Loads saved model + test set, generates full report, confusion matrix, and plots.
"""

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.sparse as sp
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, top_k_accuracy_score

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

TARGET_COL = "Podrijetlo imena"
TEXT_COL = "combined_text"


# ---------------------------------------------------------------------------
# Load artifacts & test set
# ---------------------------------------------------------------------------

def load_artifacts():
    model      = joblib.load(MODELS_DIR / "best_model.joblib")
    tfidf_char = joblib.load(MODELS_DIR / "tfidf_char.joblib")
    tfidf_word = joblib.load(MODELS_DIR / "tfidf_word.joblib")
    le_target  = joblib.load(MODELS_DIR / "label_encoder.joblib")
    svd = joblib.load(MODELS_DIR / "svd.joblib") if (MODELS_DIR / "svd.joblib").exists() else None
    with open(MODELS_DIR / "feature_config.json", encoding="utf-8") as f:
        config = json.load(f)
    return model, tfidf_char, tfidf_word, svd, le_target, config


def build_test_features(df_test, tfidf_char, tfidf_word, svd, config):
    handcrafted = config["handcrafted_features"]
    best_fs  = config["best_feature_set"]
    uses_svd = config.get("uses_svd", False)

    X_A = df_test[handcrafted].values.astype(float)
    texts = df_test[TEXT_COL].fillna("").tolist()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_char = tfidf_char.transform(texts)
        X_word = tfidf_word.transform(texts)

    if best_fs == "A_handcrafted":
        return X_A
    elif uses_svd and svd is not None:
        # HistGB with SVD-reduced dense features
        X_B_sparse = sp.hstack([X_char, X_word], format="csr")
        X_B_dense  = svd.transform(X_B_sparse)
        if best_fs == "B_tfidf":
            return X_B_dense
        else:  # C_combined
            return np.hstack([X_A, X_B_dense])
    elif best_fs == "B_tfidf":
        return sp.hstack([X_char, X_word], format="csr")
    else:  # C_combined
        return sp.hstack([sp.csr_matrix(X_A), X_char, X_word], format="csr")


# ---------------------------------------------------------------------------
# 1. Classification report
# ---------------------------------------------------------------------------

def section_classification_report(model, X_test, y_test, classes, lines):
    y_pred = model.predict(X_test)
    report = classification_report(y_test, y_pred, target_names=classes, digits=4)
    print("\n--- Classification Report ---")
    print(report)
    lines += ["CLASSIFICATION REPORT", "=" * 60, report]
    return y_pred


# ---------------------------------------------------------------------------
# 2. Confusion matrix
# ---------------------------------------------------------------------------

def section_confusion_matrix(y_test, y_pred, classes, lines):
    cm = confusion_matrix(y_test, y_pred)

    # Text table
    col_w = 10
    header = " " * 36 + "".join(f"{c[:col_w]:>{col_w}}" for c in classes)
    rows = [header]
    for i, cls in enumerate(classes):
        row = f"{cls[:35]:>35} " + "".join(f"{cm[i, j]:>{col_w}}" for j in range(len(classes)))
        rows.append(row)
    cm_text = "\n".join(rows)

    print("\n--- Confusion Matrix ---")
    print(cm_text)

    with open(REPORTS_DIR / "confusion_matrix.txt", "w", encoding="utf-8") as f:
        f.write("CONFUSION MATRIX\n\n" + cm_text + "\n")

    lines += ["\nCONFUSION MATRIX", cm_text]

    # Heatmap PNG
    short = [c[:22] for c in classes]
    fig, ax = plt.subplots(figsize=(11, 8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=short, yticklabels=short, ax=ax, linewidths=0.5,
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — Best Model", fontsize=14)
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close()
    print(f"Saved reports/confusion_matrix.png")

    return cm


# ---------------------------------------------------------------------------
# 3. Per-class top-k analysis
# ---------------------------------------------------------------------------

def section_per_class_topk(model, X_test, y_test, classes, lines):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    print("\n--- Per-class Top-k Accuracy ---")
    lines.append("\nPER-CLASS TOP-K ACCURACY")
    for i, cls in enumerate(classes):
        mask = y_test == i
        if not mask.any():
            continue
        top1 = (y_pred[mask] == i).mean()
        top2 = np.any(np.argsort(y_proba[mask], axis=1)[:, -2:] == i, axis=1).mean()
        top3 = np.any(np.argsort(y_proba[mask], axis=1)[:, -3:] == i, axis=1).mean()
        line = f"  {cls:<35}  n={mask.sum():>4}  top-1={top1:.3f}  top-2={top2:.3f}  top-3={top3:.3f}"
        print(line)
        lines.append(line)

    ov1 = top_k_accuracy_score(y_test, y_proba, k=1)
    ov2 = top_k_accuracy_score(y_test, y_proba, k=2)
    ov3 = top_k_accuracy_score(y_test, y_proba, k=3)
    overall = f"\n  Overall  top-1={ov1:.4f}  top-2={ov2:.4f}  top-3={ov3:.4f}"
    print(overall)
    lines.append(overall)

    # Most confused pairs
    cm_no_diag = confusion_matrix(y_test, y_pred)
    np.fill_diagonal(cm_no_diag, 0)
    confused = [
        (cm_no_diag[i, j], classes[i], classes[j])
        for i in range(len(classes))
        for j in range(len(classes))
        if i != j and cm_no_diag[i, j] > 0
    ]
    print("\n  Top confused pairs (true -> predicted):")
    lines.append("\n  Top confused pairs:")
    for count, true_cls, pred_cls in sorted(confused, reverse=True)[:6]:
        line = f"    {true_cls}  ->  {pred_cls}: {count}"
        print(line)
        lines.append(line)


# ---------------------------------------------------------------------------
# 4. Confidence analysis
# ---------------------------------------------------------------------------

def section_confidence(model, X_test, y_test, classes, lines):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    confidence = y_proba.max(axis=1)
    correct = y_pred == y_test

    print("\n--- Confidence Analysis ---")
    print(f"  Mean confidence (correct):   {confidence[correct].mean():.4f}")
    print(f"  Mean confidence (incorrect): {confidence[~correct].mean():.4f}")
    print("\n  Threshold | Precision | Coverage")

    best_threshold = 0.5
    for t in np.arange(0.3, 1.0, 0.05):
        mask = confidence >= t
        if not mask.any():
            continue
        prec = correct[mask].mean()
        cov  = mask.mean()
        flag = ""
        if prec >= 0.85 and best_threshold == 0.5:
            best_threshold = t
            flag = "  <-- suggested threshold"
        print(f"    {t:.2f}    |   {prec:.3f}   |  {cov:.2%}{flag}")

    lines += [
        "\nCONFIDENCE ANALYSIS",
        f"Mean confidence (correct):   {confidence[correct].mean():.4f}",
        f"Mean confidence (incorrect): {confidence[~correct].mean():.4f}",
        f"Suggested threshold for human review flag: {best_threshold:.2f}",
    ]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(confidence[correct],  bins=30, alpha=0.7, label="Correct",   color="steelblue")
    ax.hist(confidence[~correct], bins=30, alpha=0.7, label="Incorrect", color="tomato")
    ax.axvline(best_threshold, color="black", linestyle="--", label=f"Threshold={best_threshold:.2f}")
    ax.set_xlabel("Predicted Probability (Confidence)", fontsize=12)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("Confidence Distribution: Correct vs Incorrect Predictions", fontsize=13)
    ax.legend()
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confidence_distribution.png", dpi=150)
    plt.close()
    print(f"Saved reports/confidence_distribution.png")


# ---------------------------------------------------------------------------
# 5. Error analysis
# ---------------------------------------------------------------------------

def section_error_analysis(model, X_test, y_test, df_test, classes, lines):
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    error_idx = np.where(y_pred != y_test)[0]
    print(f"\n--- Error Analysis ({len(error_idx)} total errors) ---")
    lines.append(f"\nERROR ANALYSIS ({len(error_idx)} total errors — 20 shown, minority classes first)")

    MINORITY = {"preuzeto iz literature", "preuzeto sa karte", "nepoznato podrijetlo"}
    minority_idx = [i for i in error_idx if classes[y_test[i]] in MINORITY]
    majority_idx = [i for i in error_idx if classes[y_test[i]] not in MINORITY]
    show_idx = (minority_idx + majority_idx)[:20]

    df_reset = df_test.reset_index(drop=True)
    for rank, i in enumerate(show_idx, 1):
        name     = df_reset.iloc[i].get("Ime objekta", "?")
        true_cls = classes[y_test[i]]
        pred_cls = classes[y_pred[i]]
        top3     = np.argsort(y_proba[i])[::-1][:3]
        top3_str = "  |  ".join(f"{classes[j]} ({y_proba[i][j]:.2f})" for j in top3)
        line = f"  [{rank:02d}] '{name}'  |  true={true_cls}  |  pred={pred_cls}  |  top3: {top3_str}"
        print(line)
        lines.append(line)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    print("Loading model and artifacts ...")
    model, tfidf_char, tfidf_word, svd, le_target, config = load_artifacts()
    classes     = config["classes"]
    handcrafted = config["handcrafted_features"]
    best_fs     = config["best_feature_set"]
    best_model  = config.get("best_model", "")
    print(f"Best model: {best_model} / {best_fs}")

    test_csv = DATA_DIR / "test_set.csv"
    print(f"Loading test set from {test_csv.name} ...")
    df_test = pd.read_csv(test_csv, encoding="utf-8-sig", dtype=str)
    y_test = df_test["_y_true"].astype(int).values
    df_test = df_test.drop(columns=["_y_true"])

    for col in handcrafted:
        df_test[col] = pd.to_numeric(df_test[col], errors="coerce").fillna(0)

    print("Building test feature matrix ...")
    X_test = build_test_features(df_test, tfidf_char, tfidf_word, svd, config)

    report_lines = [
        "EVALUATION REPORT",
        "=" * 60,
        f"Best model: {best_model} / {best_fs}",
        f"Test samples: {len(y_test)}",
        "",
    ]

    section_classification_report(model, X_test, y_test, classes, report_lines)
    section_confusion_matrix(y_test, model.predict(X_test), classes, report_lines)
    section_per_class_topk(model, X_test, y_test, classes, report_lines)
    section_confidence(model, X_test, y_test, classes, report_lines)
    section_error_analysis(model, X_test, y_test, df_test, classes, report_lines)

    report_path = REPORTS_DIR / "evaluation_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(report_lines))
    print(f"\nSaved full report: {report_path}")
    print("\nEvaluation complete.")


if __name__ == "__main__":
    run()
