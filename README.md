[README.md](https://github.com/user-attachments/files/30457392/README.md)
# Starting With What You Have

Code and data for *Starting With What You Have: Quantitative and Spatial Methods for Urban Research in the Global South* by Joolim Lee, PhD.

Twelve methods, and the data conditions each one actually needs. Every method in Part 2 of the book is runnable here, on synthetic data with a known answer planted in it, so you can confirm the method works before trusting it on your own material.

## Quick start

No installation, in a browser:

1. Open any notebook under `chNN-*/notebooks/` in [Google Colab](https://colab.research.google.com/).
2. Run all cells.

Locally:

```bash
git clone https://github.com/USER/starting-with-what-you-have.git
cd starting-with-what-you-have
pip install -r requirements.txt
python ch04-spatial-statistics/spatial_hotspots.py
```

## What is here

| Directory | Chapter | Method |
|---|---|---|
| `ch04-spatial-statistics/` | 4 | Moran's I, LISA, Getis-Ord Gi* |
| `ch05-gwr-mgwr/` | 5 | Geographically weighted regression |
| `ch06-fsqca/` | 6 | Fuzzy-set QCA (first Python implementation) |
| `ch07-remote-sensing/` | 7 | Spectral indices, classification, land surface temperature |
| `ch08-vgi-osm/` | 8 | OpenStreetMap completeness and bias correction |
| `ch09-ahp-family/` | 9 | AHP, ANP, fuzzy AHP, IPA |
| `ch10-spatial-ml-xai/` | 10 | Spatial block CV, SHAP, partial dependence |
| `ch11-text-and-image/` | 11 | Text classification, topic models, Green View Index |
| `shared/` | | Plotting style used by every figure |
| `docs/` | | Reproduction notes and verification status |

Chapters 1 to 3 and 12 to 17 are argument rather than procedure and have no code.

## About the sample data

Every dataset here is **synthetic, with the correct answer planted in advance**. That is deliberate. It lets you verify that a method recovers a structure you already know about before you apply it to data where you do not. Each chapter README states what was planted and what the analysis should return.

It also means no real city's data is redistributed here, so there are no data licensing constraints on reuse.

## Verifying the manuscript

Three scripts in the repository root check different things:

```bash
python verify_book_numbers.py    # E1: every number in the book against a fresh run
python verify_content.py         # cross-references, citations, numerical consistency
python verify_content_pass2.py   # overreach, contradictions, missing causal caveats
```

The last two check the manuscript text itself, which is not included in this code repository. They are here for anyone maintaining a fork that keeps the prose alongside the code.

## Reproducing the book's numbers

Every figure quoted in the book comes from this code. Random seeds are fixed and stated in each module. If a number you get differs from the book, check the seed first, then the package versions in `requirements.txt`.

## Verification status

Results are reproduced by running the code. Cross-validation against established reference software is in progress and tracked in `docs/verification.md`. Notably, `ch06-fsqca/fsqca.py` has not yet been checked against fsQCA 4.x or R's `QCA` package; the book says so, and this repository will carry the comparison when it is done.

## A note on how this was built

The code in this repository was developed with [Claude](https://claude.ai) (Anthropic). The analysis choices, the structure, and the argument are the author's. See the AI use declaration in the book's front matter.

## Licence

Code is MIT. Synthetic data is CC0 1.0. You may use, modify, and redistribute both, including commercially, without asking.

If you use this material in published work, a citation is appreciated but not required:

> Lee, J. (2026). *Starting With What You Have: Quantitative and Spatial Methods for Urban Research in the Global South*.
