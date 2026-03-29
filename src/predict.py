"""
Inference interface for the trained name-origin classifier.
Loads saved model artifacts and predicts Podrijetlo imena for new entries.
"""

import json
import re
import sys
import warnings
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import scipy.sparse as sp

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"

GENERIC_PREFIXES = {"jama", "špilja", "spilja", "ponor", "estavela", "kaverna", "ledenica"}
PREPOSITIONS     = {"kod", "u", "na", "pod", "iznad", "ispod", "kraj", "blizu", "do", "vrh", "pokraj", "između"}
POSSESSIVE_SUFFIXES = ("ova", "eva", "ina", "ića", "ica", "in", "ska", "ška")
DESCRIPTIVE_WORDS   = {"velika", "mala", "gornja", "donja", "stara", "nova", "crna", "bijela", "suha", "mokra"}


def _tokenize(text: str) -> set:
    if not text:
        return set()
    return set(re.findall(r"[a-zA-ZčćšžđČĆŠŽĐ]+", text.lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


class NameOriginPredictor:
    """Load saved model artifacts and predict cave name origin."""

    def __init__(self, models_dir: Optional[Path] = None):
        d = Path(models_dir) if models_dir else MODELS_DIR
        self.model      = joblib.load(d / "best_model.joblib")
        self.tfidf_char = joblib.load(d / "tfidf_char.joblib")
        self.tfidf_word = joblib.load(d / "tfidf_word.joblib")
        self.le_target  = joblib.load(d / "label_encoder.joblib")
        self.vrsta_enc  = joblib.load(d / "vrsta_encoder.joblib")
        self.svd = joblib.load(d / "svd.joblib") if (d / "svd.joblib").exists() else None
        with open(d / "feature_config.json", encoding="utf-8") as f:
            self.config = json.load(f)
        self.classes  = self.config["classes"]
        self.best_fs  = self.config["best_feature_set"]
        self.uses_svd = self.config.get("uses_svd", False)

    def _handcrafted(self, name: str, sinonimi: str, lokalitet: str, mjesto: str, vrsta: str) -> np.ndarray:
        name_tok = _tokenize(name)
        lok_tok  = _tokenize(lokalitet)
        mj_tok   = _tokenize(mjesto)
        words    = name.split()
        alpha    = [c for c in name if c.isalpha()]
        sin_list = [s.strip() for s in sinonimi.split(",") if s.strip()] if sinonimi else []

        try:
            vrsta_enc = self.vrsta_enc.transform([vrsta or "unknown"])[0]
        except ValueError:
            vrsta_enc = -1

        return np.array([
            len(words),
            len(name),
            int(any(c.isdigit() for c in name)),
            int(bool(name_tok & GENERIC_PREFIXES)),
            int(bool(name_tok & PREPOSITIONS)),
            sum(1 for c in alpha if c.isupper()) / len(alpha) if alpha else 0.0,
            int(len(words) == 1),
            int(any(words[-1].lower().endswith(s) for s in POSSESSIVE_SUFFIXES) if words else False),
            _jaccard(name_tok, lok_tok),
            _jaccard(name_tok, mj_tok),
            int(bool({t for t in name_tok if len(t) >= 4} & lok_tok)),
            int(bool({t for t in name_tok if len(t) >= 4} & mj_tok)),
            int(bool(sinonimi and sinonimi.strip())),
            len(sin_list),
            vrsta_enc,
            int(bool(name_tok & DESCRIPTIVE_WORDS)),
            int(
                len(words) == 1
                and not (name_tok & GENERIC_PREFIXES)
                and not (name_tok & PREPOSITIONS)
                and not (name_tok & DESCRIPTIVE_WORDS)
                and not (name_tok & lok_tok)
                and not (name_tok & mj_tok)
            ),
        ], dtype=float)

    def predict(
        self,
        name: str,
        sinonimi: str = "",
        lokalitet: str = "",
        mjesto: str = "",
        vrsta: str = "",
    ) -> dict:
        """
        Predict origin class for a single cave entry.

        Returns:
            dict: prediction (str), confidence (float), probabilities (dict[str, float])
        """
        combined = f"{name} {sinonimi} {lokalitet} {mjesto}".strip()
        X_A = self._handcrafted(name, sinonimi, lokalitet, mjesto, vrsta).reshape(1, -1)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_char = self.tfidf_char.transform([combined])
            X_word = self.tfidf_word.transform([combined])

        if self.best_fs == "A_handcrafted":
            X = X_A
        elif self.uses_svd and self.svd is not None:
            X_B_sparse = sp.hstack([X_char, X_word], format="csr")
            X_B_dense  = self.svd.transform(X_B_sparse)
            X = X_B_dense if self.best_fs == "B_tfidf" else np.hstack([X_A, X_B_dense])
        elif self.best_fs == "B_tfidf":
            X = sp.hstack([X_char, X_word], format="csr")
        else:
            X = sp.hstack([sp.csr_matrix(X_A), X_char, X_word], format="csr")

        proba    = self.model.predict_proba(X)[0]
        pred_idx = proba.argmax()
        return {
            "prediction":    self.classes[pred_idx],
            "confidence":    float(proba[pred_idx]),
            "probabilities": {cls: float(p) for cls, p in zip(self.classes, proba)},
        }


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

def run_demo():
    print("Loading model ...")
    predictor = NameOriginPredictor()

    examples = [
        {"name": "Jama kod Munja",        "lokalitet": "Munjan",  "mjesto": "Munjan"},
        {"name": "Lukina jama",            "lokalitet": "Velebit", "mjesto": "Baške Oštarije"},
        {"name": "Vilinska špilja",        "lokalitet": "",        "mjesto": "Omiš"},
        {"name": "Ponor Malog Rujna",      "lokalitet": "Malo Rujno", "mjesto": "Knin"},
        {"name": "Špilja kod Izvora Rupe", "lokalitet": "Rupa",    "mjesto": "Rupa"},
    ]

    for ex in examples:
        r = predictor.predict(**ex)
        print(f"\nName: {ex['name']}")
        print(f"  → {r['prediction']}  (confidence: {r['confidence']:.3f})")
        for cls, p in sorted(r["probabilities"].items(), key=lambda x: -x[1]):
            bar = "█" * int(p * 30)
            print(f"     {cls:<35} {p:.4f}  {bar}")


if __name__ == "__main__":
    run_demo()
