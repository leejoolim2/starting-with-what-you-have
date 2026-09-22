# Chapter 8. Volunteered geographic information

OSM completeness estimation, bias correction, and street orientation entropy.

## Run it

```bash
pip install -r ../requirements.txt
python vgi_toolkit.py
```

Or open `notebooks/` in Google Colab, which needs no installation.

## Expected result

Naive OSM inverts the density ranking; contributor count predicts completeness at r = 0.959.

## Contents

| Path | What it is |
|---|---|
| `*.py` | Analysis module used by the chapter |
| `notebooks/` | Colab-ready walkthrough |
| `data/` | Synthetic sample data with a known answer planted in it |
| `figures/` | The figure printed in the book |

## Licence

Code: MIT. Data: CC0 1.0.
