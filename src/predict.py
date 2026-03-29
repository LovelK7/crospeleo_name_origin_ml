"""
Prediction interface for the CroSpeleo name-origin classifier.

Single prediction:
    python src/predict.py --name "Jama pod Vrsicen" --lokalitet "Vrsic" --mjesto "Kranjska Gora"

Batch prediction:
    python src/predict.py --batch data/processed/new_objects.csv --output data/processed/predictions.csv
"""

import argparse
import json
import re
import sys
import warnings
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

import joblib
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).parent.parent
MODELS_DIR = ROOT / "models"

# --- Vocabulary (mirrors feature_engineering.py) ---
GENERIC_PREFIXES  = {"jama", "špilja", "spilja", "ponor", "estavela", "kaverna", "ledenica"}
PREPOSITIONS      = {"kod", "u", "na", "pod", "iznad", "ispod", "kraj", "blizu", "do", "vrh", "pokraj", "između"}
POSSESSIVE_SFXS   = ("ova", "eva", "ina", "ića", "ica", "in", "ska", "ška")
DESCRIPTIVE_WORDS = {"velika", "mala", "gornja", "donja", "stara", "nova", "crna", "bijela", "suha", "mokra"}

# Classes where we flag for human review unless very confident
HARD_CLASSES          = {"preuzeto kao lokalni naziv", "preuzeto sa karte", "preuzeto iz literature"}
CONFIRMATION_THRESHOLD = 0.6   # flag if confidence below this
HARD_CLASS_THRESHOLD   = 0.8   # flag hard classes unless above this


def _tok(text: str) -> set:
    if not text:
        return set()
    return set(re.findall(r"[a-zA-ZčćšžđČĆŠŽĐ]+", text.lower()))


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


# ---------------------------------------------------------------------------
# Main predictor class
# ---------------------------------------------------------------------------

class OriginPredictor:
    """Load saved model artifacts and predict cave-name origin with explanation."""

    def __init__(self, model_dir: str = "models/"):
        d = Path(model_dir)
        self.model      = joblib.load(d / "best_model.joblib")
        self.tfidf_char = joblib.load(d / "tfidf_char.joblib")
        self.tfidf_word = joblib.load(d / "tfidf_word.joblib")
        self.svd        = joblib.load(d / "svd.joblib") if (d / "svd.joblib").exists() else None
        self.le_target  = joblib.load(d / "label_encoder.joblib")
        self.vrsta_enc  = joblib.load(d / "vrsta_encoder.joblib")
        with open(d / "feature_config.json", encoding="utf-8") as f:
            self.config = json.load(f)

        self.classes     = self.config["classes"]
        self.best_fs     = self.config["best_feature_set"]
        self.uses_svd    = self.config.get("uses_svd", False)
        self.hc_features = self.config["handcrafted_features"]

        # Nearest-neighbour lookup data
        self._train_matrix = sp.load_npz(d / "train_tfidf_matrix.npz")
        lookup_df = pd.read_csv(d / "train_lookup.csv", encoding="utf-8-sig", dtype=str)
        self._train_names  = lookup_df["Ime objekta"].fillna("").tolist()
        self._train_labels = lookup_df["Podrijetlo imena"].fillna("").tolist()

    # ------------------------------------------------------------------
    # Feature construction
    # ------------------------------------------------------------------

    def _handcrafted(self, ime: str, sinonimi: str, lokalitet: str, mjesto: str, vrsta: str) -> np.ndarray:
        name_tok = _tok(ime)
        lok_tok  = _tok(lokalitet)
        mj_tok   = _tok(mjesto)
        words    = ime.split()
        alpha    = [c for c in ime if c.isalpha()]
        sin_list = [s.strip() for s in sinonimi.split(",") if s.strip()] if sinonimi else []

        try:
            vrsta_enc = self.vrsta_enc.transform([vrsta or "unknown"])[0]
        except ValueError:
            vrsta_enc = -1

        return np.array([
            len(words),
            len(ime),
            int(any(c.isdigit() for c in ime)),
            int(bool(name_tok & GENERIC_PREFIXES)),
            int(bool(name_tok & PREPOSITIONS)),
            sum(1 for c in alpha if c.isupper()) / len(alpha) if alpha else 0.0,
            int(len(words) == 1),
            int(any(words[-1].lower().endswith(s) for s in POSSESSIVE_SFXS) if words else False),
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

    def _build_X(self, combined_text: str, X_A: np.ndarray):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            X_char = self.tfidf_char.transform([combined_text])
            X_word = self.tfidf_word.transform([combined_text])

        if self.best_fs == "A_handcrafted":
            return X_A.reshape(1, -1)
        elif self.uses_svd and self.svd is not None:
            X_B = sp.hstack([X_char, X_word], format="csr")
            X_B_dense = self.svd.transform(X_B)
            if self.best_fs == "B_tfidf":
                return X_B_dense
            return np.hstack([X_A.reshape(1, -1), X_B_dense])
        elif self.best_fs == "B_tfidf":
            return sp.hstack([X_char, X_word], format="csr")
        else:  # C_combined
            return sp.hstack([sp.csr_matrix(X_A.reshape(1, -1)), X_char, X_word], format="csr")

    # ------------------------------------------------------------------
    # Nearest-neighbour similarity
    # ------------------------------------------------------------------

    def _find_similar(self, combined_text: str, n: int = 5) -> list:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            x_vec = self.tfidf_char.transform([combined_text])
        sims = cosine_similarity(x_vec, self._train_matrix).flatten()
        top_idx = np.argsort(sims)[::-1][:n]
        return [
            {
                "name":       self._train_names[i],
                "origin":     self._train_labels[i],
                "similarity": round(float(sims[i]), 4),
            }
            for i in top_idx
        ]

    # ------------------------------------------------------------------
    # Explanation
    # ------------------------------------------------------------------

    def _explain(
        self,
        ime: str,
        sinonimi: str,
        lokalitet: str,
        X_A: np.ndarray,
        predicted_class: str,
        confidence: float,
    ) -> list:
        hc = dict(zip(self.hc_features, X_A))
        reasons = []

        # Preposition signal
        if hc.get("name_has_preposition", 0):
            preps = [w for w in _tok(ime) if w in PREPOSITIONS]
            if preps:
                reasons.append(f"naziv sadrži prijedlog '{preps[0]}' — upućuje na toponimsko podrijetlo")

        # Lokalitet overlap
        if hc.get("name_equals_lokalitet_token", 0) or hc.get("name_lokalitet_overlap", 0) > 0.25:
            reasons.append("naziv se poklapa s lokalitetom")

        # Mjesto overlap
        if hc.get("name_equals_mjesto_token", 0) and "lokalitet" not in [r[:10] for r in reasons]:
            reasons.append("naziv se poklapa s najbližim mjestom")

        # Single creative word
        if hc.get("name_is_single_word", 0) and hc.get("name_looks_humorous_or_creative", 0):
            reasons.append("naziv je jednočlani negeografski izraz — upućuje na smišljeno novo ime")

        # Possessive suffix
        if hc.get("name_has_possessive_suffix", 0):
            reasons.append("naziv ima posesivni sufiks — upućuje na toponimsko podrijetlo")

        # Synonym exists
        if hc.get("sinonimi_exists", 0):
            reasons.append("sinonim postoji — moguće preuzeto iz literature ili s karte")

        # Descriptive word
        if hc.get("name_looks_descriptive", 0):
            reasons.append("naziv sadrži opisnu odredbu (velika/mala/gornja...)")

        # Number in name
        if hc.get("name_has_number", 0):
            reasons.append("naziv sadrži broj — vjerojatno smišljeno novo ime")

        # Fallback
        if not reasons or confidence < 0.4:
            reasons.append("nema jasnih signala — preporuča se ručna provjera")

        return reasons[:3]

    # ------------------------------------------------------------------
    # Public predict()
    # ------------------------------------------------------------------

    def predict(
        self,
        ime_objekta: str,
        lokalitet: str = "",
        najblize_mjesto: str = "",
        sinonimi: str = "",
        vrsta_objekta: str = "",
        napomena: str = "",
    ) -> dict:
        ime  = ime_objekta or ""
        lok  = lokalitet or ""
        mj   = najblize_mjesto or ""
        sin  = sinonimi or ""
        vrst = vrsta_objekta or ""

        combined_text = f"{ime} {sin} {lok} {mj}".strip()
        X_A = self._handcrafted(ime, sin, lok, mj, vrst)
        X   = self._build_X(combined_text, X_A)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            proba = self.model.predict_proba(X)[0]

        # Sort all 6 classes by probability
        order      = np.argsort(proba)[::-1]
        top_k      = [{"value": self.classes[i], "score": round(float(proba[i]), 4)} for i in order]
        pred_class = top_k[0]["value"]
        confidence = top_k[0]["score"]

        # needs_user_confirmation logic
        needs_confirm = confidence < CONFIRMATION_THRESHOLD
        if pred_class in HARD_CLASSES and confidence <= HARD_CLASS_THRESHOLD:
            needs_confirm = True

        explanation     = self._explain(ime, sin, lok, X_A, pred_class, confidence)
        similar_examples = self._find_similar(combined_text)

        return {
            "predicted_value":        pred_class,
            "confidence":             round(confidence, 4),
            "top_k":                  top_k,
            "explanation":            explanation,
            "similar_examples":       similar_examples,
            "needs_user_confirmation": needs_confirm,
        }

    # ------------------------------------------------------------------
    # Batch prediction
    # ------------------------------------------------------------------

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        results = []
        for _, row in df.iterrows():
            r = self.predict(
                ime_objekta    = str(row.get("Ime objekta", "") or ""),
                lokalitet      = str(row.get("Lokalitet", "") or ""),
                najblize_mjesto= str(row.get("Najbliže mjesto", "") or ""),
                sinonimi       = str(row.get("Sinonimi", "") or ""),
                vrsta_objekta  = str(row.get("Vrsta objekta", "") or ""),
                napomena       = str(row.get("Napomena (osnovni podaci)", "") or ""),
            )
            top = r["top_k"]
            results.append({
                "predicted_origin":  r["predicted_value"],
                "confidence":        r["confidence"],
                "top_2":             top[1]["value"] if len(top) > 1 else "",
                "top_3":             top[2]["value"] if len(top) > 2 else "",
                "needs_confirmation": r["needs_user_confirmation"],
            })
        return pd.concat([df.reset_index(drop=True), pd.DataFrame(results)], axis=1)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _pretty_print(result: dict) -> None:
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description="CroSpeleo name-origin predictor")
    parser.add_argument("--name",      default="", help="Ime objekta")
    parser.add_argument("--lokalitet", default="", help="Lokalitet")
    parser.add_argument("--mjesto",    default="", help="Najbliže mjesto")
    parser.add_argument("--sinonimi",  default="", help="Sinonimi")
    parser.add_argument("--vrsta",     default="", help="Vrsta objekta")
    parser.add_argument("--napomena",  default="", help="Napomena")
    parser.add_argument("--batch",     default="", help="Input CSV for batch mode")
    parser.add_argument("--output",    default="", help="Output CSV for batch mode")
    parser.add_argument("--model-dir", default="models/", help="Path to models directory")
    args = parser.parse_args()

    predictor = OriginPredictor(model_dir=args.model_dir)

    if args.batch:
        batch_path = Path(args.batch)
        print(f"Batch mode: {batch_path.name}")
        df = pd.read_csv(batch_path, encoding="utf-8-sig", dtype=str).fillna("")
        out_df = predictor.predict_batch(df)
        out_path = Path(args.output) if args.output else batch_path.with_name("predictions.csv")
        out_df.to_csv(out_path, index=False, encoding="utf-8-sig")
        print(f"Saved {len(out_df)} predictions to {out_path}")
    elif args.name:
        result = predictor.predict(
            ime_objekta    = args.name,
            lokalitet      = args.lokalitet,
            najblize_mjesto= args.mjesto,
            sinonimi       = args.sinonimi,
            vrsta_objekta  = args.vrsta,
            napomena       = args.napomena,
        )
        _pretty_print(result)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
