# Predictive Maintenance Classifier

A binary classifier that predicts industrial machine failures from sensor readings, using the UCI AI4I 2020 dataset. Built as a reference implementation for the AMCS AI Engineer prep work.

## What this exercises

- sklearn `Pipeline` + `ColumnTransformer` for clean train/test handling
- `GridSearchCV` for hyperparameter search with cross-validation
- Class imbalance handling (~3.4% positive rate) — uses `average_precision` as the search metric and tunes the decision threshold post-fit
- Proper evaluation: ROC-AUC, PR-AUC, confusion matrix at tuned threshold
- A small test suite that asserts data invariants

## JD coverage

- "Proficiency in Python ... and relevant ML libraries"
- "Strong problem-solving and analytical skills"
- "Ensure good quality code with high levels of code coverage" (tests cover data invariants)

## How to run

```powershell
cd samples\easy-predictive-maintenance
py -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Trains, saves model.joblib and metrics.json.
# Dataset is auto-downloaded on first run.
python -m src.train

# Loads model.joblib, prints classification_report, writes ROC + PR curve PNGs.
python -m src.evaluate

# Run tests
pytest tests/
```

## Expected output

After training:

```
=== Results ===
{
  "roc_auc": 0.97x,
  "pr_auc": 0.7xx-0.8xx,
  "best_params": {...},
  "threshold": 0.2x-0.4x,
  "at_threshold": {
    "f1": 0.6x-0.7x,
    "confusion_matrix": {"tn": ~1930, "fp": ~10-25, "fn": ~15-25, "tp": ~45-55}
  },
  "positive_rate_test": 0.0339
}
```

The exact numbers vary with the random seed and the CV-picked hyperparameters, but ROC-AUC should land around 0.97. Notice that PR-AUC is much lower than ROC-AUC — that's the imbalance talking. **For an imbalanced problem like this, PR-AUC is the metric to optimize, not accuracy or ROC-AUC.** That insight alone is interview gold.

## File layout

```
src/
├── data.py         # download + load + drop ID cols + stratified split
├── features.py     # ColumnTransformer (StandardScaler + OneHotEncoder)
├── train.py        # GridSearchCV → fit → threshold tune → save artifacts
└── evaluate.py     # load model → classification report → curve PNGs

tests/
└── test_data.py    # invariants on the loaded data + split

data/
└── README.md       # dataset description + manual download instructions
```

## What to try next

1. **Swap the classifier**: replace `GradientBoostingClassifier` with `RandomForestClassifier` or `HistGradientBoostingClassifier` and compare.
2. **Add SMOTE**: install `imbalanced-learn` and put `SMOTE` in the pipeline. Compare metrics — does it help on this dataset?
3. **Per-failure-mode breakdown**: the dataset has five failure-mode flags (`TWF`, `HWF`, `PWF`, `OSF`, `RNF`) that we currently drop. Turn the problem into multi-label classification.
4. **Connect to prep project #4**: feed the trained model into Fairlearn and audit predictions across the `Type` feature (L/M/H) — see whether the model is biased against any machine class.
