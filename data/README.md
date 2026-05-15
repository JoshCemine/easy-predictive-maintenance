# Dataset: UCI AI4I 2020 Predictive Maintenance

**Source:** https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset

**Direct CSV:** https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip

**Shape:** 10,000 rows × 14 columns
**Target:** `Machine failure` (binary, ~3.4% positive)
**Features used:** Type, Air temperature, Process temperature, Rotational speed, Torque, Tool wear

## Download

The training script will auto-download if `ai4i2020.csv` is not present in this directory. To download manually:

```powershell
Invoke-WebRequest -Uri "https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv" -OutFile "ai4i2020.csv"
```

## Why this dataset

Real industrial sensor data, well-documented, no licensing friction, and the class imbalance (~3% positive) forces you to think about precision-recall tradeoffs instead of just optimizing accuracy. This mirrors the kind of failure-prediction problem AMCS customers actually have on waste-collection fleets and recycling-facility equipment.
