"""Load and clean the CroSpeleo source Excel file."""

from pathlib import Path

import pandas as pd

from src.utils import ensure_dir

# Column indices (0-based) → positions 4,6,7,16,17,35,47 in the spreadsheet
# openpyxl usecols uses 0-based integers
USECOLS = [3, 5, 6, 15, 16, 34, 46]  # 0-based: col4=idx3, col6=idx5, ...

COLUMN_NAMES = [
    "Ime objekta",           # col 4  — primary input (cave name)
    "Podrijetlo imena",      # col 6  — target label
    "Sinonimi",              # col 7  — input feature
    "Najbliže mjesto",       # col 16 — input feature
    "Lokalitet",             # col 17 — input feature
    "Vrsta objekta",         # col 35 — input feature
    "Napomena (osnovni podaci)",  # col 47 — input feature
]

TARGET_COL = "Podrijetlo imena"

EXPECTED_CLASSES = {
    "smišljeno novo",
    "smišljeno prema toponimu",
    "preuzeto kao lokalni naziv",
    "preuzeto iz literature",
    "preuzeto sa karte",
    "nepoznato podrijetlo",
}


def load_raw(xlsx_path: str | Path) -> pd.DataFrame:
    xlsx_path = Path(xlsx_path)
    print(f"Loading {xlsx_path.name} ...")
    df = pd.read_excel(
        xlsx_path,
        sheet_name="Objekti",
        header=0,
        usecols=USECOLS,
        engine="openpyxl",
        dtype=str,          # read everything as string; avoids mixed-type issues
    )
    # The usecols order matches the sorted column index order, rename to Croatian names
    df.columns = COLUMN_NAMES
    print(f"  Raw rows loaded: {len(df):,}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    # Strip whitespace from all string columns
    for col in df.columns:
        df[col] = df[col].str.strip()

    # Drop rows where the target is missing or not a known class
    before = len(df)
    df = df[df[TARGET_COL].notna() & (df[TARGET_COL] != "")]
    df = df[df[TARGET_COL].isin(EXPECTED_CLASSES)]
    after = len(df)
    print(f"  Dropped {before - after:,} rows with missing/invalid target; {after:,} remain")

    # Replace empty strings with NaN for consistency across feature columns
    feature_cols = [c for c in COLUMN_NAMES if c != TARGET_COL]
    for col in feature_cols:
        df[col] = df[col].replace("", pd.NA)

    df = df.reset_index(drop=True)
    return df


def print_summary(df: pd.DataFrame) -> None:
    print(f"\n--- Dataset summary ---")
    print(f"Total rows: {len(df):,}")
    print(f"\nClass distribution:")
    counts = df[TARGET_COL].value_counts()
    for label, cnt in counts.items():
        print(f"  {label:<35} {cnt:>5}  ({cnt/len(df)*100:.1f}%)")
    print(f"\nNull counts per column:")
    for col in df.columns:
        n = df[col].isna().sum()
        print(f"  {col:<40} {n:>5}")


def run(xlsx_path: str | Path, output_dir: str | Path) -> pd.DataFrame:
    output_dir = ensure_dir(output_dir)
    df = load_raw(xlsx_path)
    df = clean(df)
    print_summary(df)

    out_path = output_dir / "dataset_clean.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {out_path}")
    return df
