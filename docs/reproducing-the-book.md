# Reproducing the book's figures and numbers

## Order of operations

Chapters are independent except for one deliberate link: Chapter 7 produces land surface temperature, which Chapter 4 can then treat as an input. You can run them in any order.

## Seeds

Every module fixes its random seed at the top of the file. The seeds are:

| Chapter | Seed |
|---|---|
| 4 | 42 |
| 5 | 7 |
| 6 | 3 |
| 7 | 11 |
| 8 | 13 |
| 9 | 2 |
| 10 | 17 |
| 11 | 11 (text), 3 (images) |

Changing a seed changes the synthetic data and therefore every number downstream. If you are checking reproduction, do not change them.

## Regenerating the figures

Figures in `chNN-*/figures/` were produced with the shared style module in `shared/figstyle.py`, which sets a CJK-capable font and a common set of sizes. The module takes a language argument so that the same script emits the Korean and English editions of each figure.

## What "planted" means

Each synthetic dataset contains a structure inserted on purpose so that the analysis has a known answer to find. For example, Chapter 10's data contains an NDVI cooling effect that saturates at exactly 0.45, and the chapter reports that the model recovers 0.450. Where recovery is imperfect, the chapter says so. This is the point of the design: you can see the method working, or failing, before you trust it on data where the truth is unknown.
