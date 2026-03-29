# Prompt 1 — Data Preparation & Feature Engineering

**Copy-paste this entire prompt into Claude in VS Code as-is.**

---

## Context

I'm building a standalone Python project to predict the "Podrijetlo imena" (origin of name) field for Croatian speleological objects. This is a multi-class classification problem with 6 classes trained on ~6000 labeled examples from the CroSpeleo cadastre.

**Project location:** 
C:\Users\Lovel\VSCode\crospeleo_name_origin_ml

**Data source:** The full labeled dataset is an Excel file at:
C:\Users\Lovel\VSCode\crospeleo_name_origin_ml\data_input\CroSpeleo - objekti.xlsx

- Sheet name: `Objekti`
- Total rows: ~6370 data rows (row 1 is header)
- 61 columns total

## Target variable

Column 6: `Podrijetlo imena` — one of 6 controlled values:

| Class | Count | % |
|---|---|---|
| smišljeno novo | 2632 | 44% |
| smišljeno prema toponimu | 1703 | 28% |
| preuzeto kao lokalni naziv | 1269 | 21% |
| preuzeto iz literature | 172 | 3% |
| preuzeto sa karte | 97 | 1.6% |
| nepoznato podrijetlo | 94 | 1.6% |

**Important:** The dataset is heavily imbalanced. The bottom 3 classes together are ~6%. This must be handled carefully in training and evaluation.

## Task — Step 1: Data loading, cleaning, feature engineering

Create the following project structure and implement data preparation:

```
crospeleo-name-origin-ml/
  README.md
  requirements.txt
  .gitignore
  data/
    raw/          # symlink or copy of source xlsx goes here
    processed/    # cleaned CSV outputs
  src/
    __init__.py
    data_loader.py
    feature_engineering.py
    utils.py
  notebooks/
    01_eda.ipynb          # optional, for manual exploration
```

### data_loader.py

Use `openpyxl` (via pandas) to load `CroSpeleo - objekti.xlsx`, sheet `Objekti`.

Extract these columns (keep original Croatian names as-is for column headers):

| Col # | Column name | Role |
|---|---|---|
| 4 | Ime objekta | **Primary input — the cave name** |
| 6 | Podrijetlo imena | **Target label** |
| 7 | Sinonimi | Input feature |
| 16 | Najbliže mjesto | Input feature |
| 17 | Lokalitet | Input feature |
| 35 | Vrsta objekta | Input feature |
| 47 | Napomena (osnovni podaci) | Input feature (may contain hints) |

Drop rows where `Podrijetlo imena` is empty/null. Save the cleaned subset as `data/processed/dataset_clean.csv` (UTF-8 with BOM for Excel compatibility).

Print summary stats: row count, class distribution, null counts per column.

### feature_engineering.py

Create a function `build_features(df) -> pd.DataFrame` that takes the cleaned dataframe and produces a feature matrix. Implement these features:

**Text-based features on `Ime objekta`:**
1. `name_word_count` — number of words
2. `name_char_count` — character length
3. `name_has_number` — contains a digit (bool→int)
4. `name_has_generic_prefix` — starts with or contains: jama, špilja, spilja, ponor, estavela, kaverna, ledenica (bool→int). These are generic speleological object type words.
5. `name_has_preposition` — contains: kod, u, na, pod, iznad, ispod, kraj, blizu, do, vrh, pokraj, između (bool→int). These spatial prepositions are strong signals for "smišljeno prema toponimu".
6. `name_uppercase_ratio` — ratio of uppercase letters to total alpha chars
7. `name_is_single_word` — name is exactly 1 word (bool→int)
8. `name_has_possessive_suffix` — ends with common Croatian possessive patterns: -ova, -eva, -ina, -ića, -ova, -in, -ska, -ška (bool→int)

**Cross-reference features:**
9. `name_lokalitet_overlap` — Jaccard similarity of word tokens between `Ime objekta` and `Lokalitet` (0.0-1.0)
10. `name_mjesto_overlap` — Jaccard similarity of word tokens between `Ime objekta` and `Najbliže mjesto` (0.0-1.0)
11. `name_equals_lokalitet_token` — any word (len>=4) in the name appears in Lokalitet (bool→int)
12. `name_equals_mjesto_token` — any word (len>=4) in the name appears in Najbliže mjesto (bool→int)
13. `sinonimi_exists` — Sinonimi field is non-empty (bool→int)
14. `sinonimi_count` — number of synonyms (split by comma)

**Object type feature:**
15. `vrsta_objekta_encoded` — label-encode `Vrsta objekta` (use sklearn LabelEncoder, handle NaN as "unknown")

**Name structure features:**
16. `name_looks_descriptive` — name contains words like: velika, mala, gornja, donja, stara, nova, crna, bijela, suha, mokra (bool→int)
17. `name_looks_humorous_or_creative` — name is a single non-geographic word that doesn't match any generic prefix, preposition, or lokalitet token — a heuristic for creative/invented names (bool→int)

**Text features for TF-IDF (to be used in Prompt 2):**
18. `combined_text` — concatenation of: `Ime objekta` + " " + `Sinonimi` + " " + `Lokalitet` + " " + `Najbliže mjesto` (fill NaN with empty string). This column will be used for TF-IDF vectorization in the next step.

All boolean features should be int (0/1). All text should be lowercased for comparisons but keep original case in `combined_text`.

### Output

After running `data_loader.py` then `feature_engineering.py`:
- Save `data/processed/dataset_features.csv` with all original columns + engineered features
- Print feature matrix shape and a sample of 10 rows
- Print class distribution one more time to confirm

### Requirements

```
pandas>=2.0
openpyxl>=3.1
scikit-learn>=1.3
numpy>=1.24
```

### Important notes
- The Excel file is 15.7 MB — use `engine='openpyxl'` and only read the needed columns if possible for speed
- All text processing must handle Croatian characters properly (č, ć, š, ž, đ) — use UTF-8 everywhere
- Do NOT use the `Katastarski broj` or `ID objekta` columns as features — they are identifiers
- Create a `main.py` or script entry point that runs the full pipeline: load → clean → features → save
- Initialize a git repo and make an initial commit after this step