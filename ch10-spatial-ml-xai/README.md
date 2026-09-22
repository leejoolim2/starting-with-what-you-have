# Chapter 10. Spatial machine learning with explainable AI

Spatial block cross-validation, SHAP, and partial dependence.

## Run it

```bash
pip install -r ../requirements.txt
python spatial_ml.py
```

Or open `notebooks/` in Google Colab, which needs no installation.

## Expected result

Random k-fold overstates R2 by 0.250 (OLS) and 0.097 (random forest); planted NDVI threshold recovered exactly at 0.450.

## Contents

| Path | What it is |
|---|---|
| `*.py` | Analysis module used by the chapter |
| `notebooks/` | Colab-ready walkthrough |
| `data/` | Synthetic sample data with a known answer planted in it |
| `figures/` | The figure printed in the book |

## Licence

Code: MIT. Data: CC0 1.0.
