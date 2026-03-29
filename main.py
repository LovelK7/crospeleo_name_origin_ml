"""Full pipeline: load → clean → features → save."""

from pathlib import Path

from src import data_loader, feature_engineering

ROOT = Path(__file__).parent
XLSX_PATH = ROOT / "data_input" / "CroSpeleo - objekti.xlsx"
PROCESSED_DIR = ROOT / "data" / "processed"


def main() -> None:
    # Step 1: load & clean
    df_clean = data_loader.run(
        xlsx_path=XLSX_PATH,
        output_dir=PROCESSED_DIR,
    )

    # Step 2: feature engineering
    feature_engineering.run(
        clean_csv=PROCESSED_DIR / "dataset_clean.csv",
        output_dir=PROCESSED_DIR,
    )

    print("\nPipeline complete.")


if __name__ == "__main__":
    main()
