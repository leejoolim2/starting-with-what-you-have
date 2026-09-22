# Chapter 6. Fuzzy-set qualitative comparative analysis

A from-scratch fsQCA implementation. No such package exists for Python.

## Run it

```bash
pip install -r ../requirements.txt
python fsqca.py
```

Or open `notebooks/` in Google Colab, which needs no installation.

## Expected result

Both planted paths recovered by the parsimonious solution; overall consistency 0.885, coverage 0.929.

## Contents

| Path | What it is |
|---|---|
| `*.py` | Analysis module used by the chapter |
| `notebooks/` | Colab-ready walkthrough |
| `data/` | Synthetic sample data with a known answer planted in it |
| `figures/` | The figure printed in the book |

## Licence

Code: MIT. Data: CC0 1.0.
