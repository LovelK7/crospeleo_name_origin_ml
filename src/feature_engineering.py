"""Feature engineering for CroSpeleo name-origin classification."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.utils import ensure_dir, jaccard, tokenize

# --- Vocabulary constants ---

GENERIC_PREFIXES = {
    "jama", "špilja", "spilja", "ponor", "estavela", "kaverna", "ledenica",
}

PREPOSITIONS = {
    "kod", "u", "na", "pod", "iznad", "ispod", "kraj", "blizu",
    "do", "vrh", "pokraj", "između",
}

POSSESSIVE_SUFFIXES = (
    "ova", "eva", "ina", "ića", "ica", "in", "ska", "ška",
)

DESCRIPTIVE_WORDS = {
    "velika", "mala", "gornja", "donja", "stara", "nova",
    "crna", "bijela", "suha", "mokra",
}


# --- Individual feature builders ---

def _name_word_count(name: str) -> int:
    if not isinstance(name, str):
        return 0
    return len(name.split())


def _name_char_count(name: str) -> int:
    if not isinstance(name, str):
        return 0
    return len(name)


def _name_has_number(name: str) -> int:
    if not isinstance(name, str):
        return 0
    return int(any(c.isdigit() for c in name))


def _name_has_generic_prefix(name: str) -> int:
    if not isinstance(name, str):
        return 0
    tokens = tokenize(name)
    return int(bool(tokens & GENERIC_PREFIXES))


def _name_has_preposition(name: str) -> int:
    if not isinstance(name, str):
        return 0
    tokens = tokenize(name)
    return int(bool(tokens & PREPOSITIONS))


def _name_uppercase_ratio(name: str) -> float:
    if not isinstance(name, str):
        return 0.0
    alpha = [c for c in name if c.isalpha()]
    if not alpha:
        return 0.0
    return sum(1 for c in alpha if c.isupper()) / len(alpha)


def _name_is_single_word(name: str) -> int:
    if not isinstance(name, str):
        return 0
    return int(len(name.split()) == 1)


def _name_has_possessive_suffix(name: str) -> int:
    if not isinstance(name, str):
        return 0
    lower = name.lower()
    # Check last word only
    words = lower.split()
    if not words:
        return 0
    last = words[-1]
    return int(any(last.endswith(s) for s in POSSESSIVE_SUFFIXES))


def _jaccard_overlap(a: str, b: str) -> float:
    return jaccard(tokenize(a), tokenize(b))


def _token_overlap(name: str, reference: str, min_len: int = 4) -> int:
    name_tokens = {t for t in tokenize(name) if len(t) >= min_len}
    ref_tokens = tokenize(reference)
    return int(bool(name_tokens & ref_tokens))


def _sinonimi_exists(val) -> int:
    return int(isinstance(val, str) and val.strip() != "")


def _sinonimi_count(val) -> int:
    if not isinstance(val, str) or not val.strip():
        return 0
    return len([s for s in val.split(",") if s.strip()])


def _name_looks_descriptive(name: str) -> int:
    if not isinstance(name, str):
        return 0
    return int(bool(tokenize(name) & DESCRIPTIVE_WORDS))


def _name_looks_humorous_or_creative(
    name: str,
    lokalitet: str,
    mjesto: str,
) -> int:
    """
    Heuristic: single non-geographic word that isn't a generic speleological
    term, doesn't contain a spatial preposition, and doesn't overlap with
    lokalitet or nearest place. Likely an invented/creative name.
    """
    if not isinstance(name, str):
        return 0
    words = name.split()
    if len(words) != 1:
        return 0
    tokens = tokenize(name)
    if tokens & GENERIC_PREFIXES:
        return 0
    if tokens & PREPOSITIONS:
        return 0
    if tokens & DESCRIPTIVE_WORDS:
        return 0
    lok_tokens = tokenize(lokalitet)
    mjes_tokens = tokenize(mjesto)
    if tokens & lok_tokens:
        return 0
    if tokens & mjes_tokens:
        return 0
    return 1


def _combined_text(row: pd.Series) -> str:
    parts = [
        row.get("Ime objekta", ""),
        row.get("Sinonimi", ""),
        row.get("Lokalitet", ""),
        row.get("Najbliže mjesto", ""),
    ]
    return " ".join(str(p) if pd.notna(p) else "" for p in parts).strip()


# --- Main function ---

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Take the cleaned dataframe and return it with all engineered feature columns appended.
    Does NOT mutate the input dataframe.
    """
    df = df.copy()

    name = df["Ime objekta"].fillna("")
    sinonimi = df["Sinonimi"].fillna("")
    lokalitet = df["Lokalitet"].fillna("")
    mjesto = df["Najbliže mjesto"].fillna("")

    # --- Text features on Ime objekta ---
    df["name_word_count"] = name.map(_name_word_count)
    df["name_char_count"] = name.map(_name_char_count)
    df["name_has_number"] = name.map(_name_has_number)
    df["name_has_generic_prefix"] = name.map(_name_has_generic_prefix)
    df["name_has_preposition"] = name.map(_name_has_preposition)
    df["name_uppercase_ratio"] = name.map(_name_uppercase_ratio)
    df["name_is_single_word"] = name.map(_name_is_single_word)
    df["name_has_possessive_suffix"] = name.map(_name_has_possessive_suffix)

    # --- Cross-reference features ---
    df["name_lokalitet_overlap"] = [
        _jaccard_overlap(n, l) for n, l in zip(name, lokalitet)
    ]
    df["name_mjesto_overlap"] = [
        _jaccard_overlap(n, m) for n, m in zip(name, mjesto)
    ]
    df["name_equals_lokalitet_token"] = [
        _token_overlap(n, l) for n, l in zip(name, lokalitet)
    ]
    df["name_equals_mjesto_token"] = [
        _token_overlap(n, m) for n, m in zip(name, mjesto)
    ]
    df["sinonimi_exists"] = sinonimi.map(_sinonimi_exists)
    df["sinonimi_count"] = sinonimi.map(_sinonimi_count)

    # --- Object type ---
    vrsta = df["Vrsta objekta"].fillna("unknown")
    le = LabelEncoder()
    df["vrsta_objekta_encoded"] = le.fit_transform(vrsta)

    # --- Name structure ---
    df["name_looks_descriptive"] = name.map(_name_looks_descriptive)
    df["name_looks_humorous_or_creative"] = [
        _name_looks_humorous_or_creative(n, l, m)
        for n, l, m in zip(name, lokalitet, mjesto)
    ]

    # --- Combined text for TF-IDF (Prompt 2) ---
    df["combined_text"] = df.apply(_combined_text, axis=1)

    return df


def run(clean_csv: str | Path, output_dir: str | Path) -> pd.DataFrame:
    clean_csv = Path(clean_csv)
    output_dir = ensure_dir(output_dir)

    print(f"Loading cleaned dataset from {clean_csv.name} ...")
    df = pd.read_csv(clean_csv, encoding="utf-8-sig", dtype=str)

    print("Building features ...")
    df_feat = build_features(df)

    print(f"\nFeature matrix shape: {df_feat.shape}")
    print(f"\nSample (10 rows, engineered columns only):")
    feature_cols = [c for c in df_feat.columns if c not in [
        "Ime objekta", "Podrijetlo imena", "Sinonimi",
        "Najbliže mjesto", "Lokalitet", "Vrsta objekta",
        "Napomena (osnovni podaci)", "combined_text",
    ]]
    print(df_feat[feature_cols].head(10).to_string())

    print(f"\nClass distribution:")
    counts = df_feat["Podrijetlo imena"].value_counts()
    for label, cnt in counts.items():
        print(f"  {label:<35} {cnt:>5}  ({cnt/len(df_feat)*100:.1f}%)")

    out_path = output_dir / "dataset_features.csv"
    df_feat.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_path}")
    return df_feat
