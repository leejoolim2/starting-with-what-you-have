# Chapter 4. Spatial descriptive statistics and hotspot analysis

Moran's I, LISA, and Getis-Ord Gi* on 144 communes.

## Run it

```bash
pip install -r ../requirements.txt
python spatial_hotspots.py
```

Or open `notebooks/` in Google Colab, which needs no installation.

## Expected result

Global Moran's I 0.706; 18 High-High units; 6 hotspots at the 95% level.

## Contents

| Path | What it is |
|---|---|
| `*.py` | Analysis module used by the chapter |
| `notebooks/` | Colab-ready walkthrough |
| `data/` | Synthetic sample data with a known answer planted in it |
| `figures/` | The figure printed in the book |

## Licence

Code: MIT. Data: CC0 1.0.
