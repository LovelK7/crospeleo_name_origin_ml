# CroSpeleo Name Origin ML

Multi-class classifier predicting the origin of names (`Podrijetlo imena`) for Croatian speleological objects.

## Classes

| Class | Description |
|---|---|
| smišljeno novo | Invented new name |
| smišljeno prema toponimu | Invented based on a toponym |
| preuzeto kao lokalni naziv | Taken as local name |
| preuzeto iz literature | Taken from literature |
| preuzeto sa karte | Taken from a map |
| nepoznato podrijetlo | Unknown origin |

## Setup

```bash
python -m venv .venv
source .venv/Scripts/activate  # Windows bash
pip install -r requirements.txt
```

## Run pipeline

```bash
python main.py
```

Outputs:
- `data/processed/dataset_clean.csv` — cleaned subset of source data
- `data/processed/dataset_features.csv` — full feature matrix ready for training

## Project structure

```
src/
  data_loader.py         # Load & clean Excel source
  feature_engineering.py # Build feature matrix
  utils.py               # Shared helpers
main.py                  # Full pipeline entry point
```
