# Chapter 5. Geographically weighted regression

GWR and MGWR on 300 synthetic parcels with a planted spatially varying coefficient.

## Run it

```bash
pip install -r ../requirements.txt
python gwr_pipeline.py
```

Or open `notebooks/` in Google Colab, which needs no installation.

## Expected result

OLS R2 0.470 -> GWR 0.694; recovery correlation 0.906; MGWR bandwidths 45 (local) vs 151 and 164 (near-global).

## Contents

| Path | What it is |
|---|---|
| `*.py` | Analysis module used by the chapter |
| `notebooks/` | Colab-ready walkthrough |
| `data/` | Synthetic sample data with a known answer planted in it |
| `figures/` | The figure printed in the book |

## Licence

Code: MIT. Data: CC0 1.0.
