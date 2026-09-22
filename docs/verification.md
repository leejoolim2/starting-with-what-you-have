# Verification status

This file records what has been checked in this repository, and what has not.

## Every number in the book reproduces

`verify_book_numbers.py` in the repository root runs each chapter's analysis from source and compares the result against the number printed in the book.

```bash
python verify_book_numbers.py
```

**Current status: 33 of 33 checks pass.**

| Chapter | Checks | Result |
|---|---|---|
| 4 | Global Moran's I, High-High count, Gi* hotspot count | pass |
| 5 | GWR coefficient recovery correlation | pass |
| 6 | Both solution paths, term count | pass |
| 7 | NDVI means, NDVI to LST correlation, warming | pass |
| 8 | Completeness by settlement type, contributor proxy correlation | pass |
| 9 | Respondents dropped, screened and unscreened weights, screening effect | pass |
| 10 | Random and spatial block CV for two models, recovered threshold, local SHAP correlation | pass |
| 11 | Vocabulary diagnostics, classification F1 by language, topic agreement, GVI recovery, feature ablation | pass |

Any failure here is a defect in the book, not in your environment. Please open an issue.

## fsQCA is cross-validated against the reference implementation

`ch06-fsqca/fsqca.py`, the from-scratch fsQCA implementation this book publishes, has been cross-validated against the reference `QCA` package in R, version 3.25.3. The comparison script is `docs/cross_validate_fsqca.R`. Agreement is exact to three decimal places on:

- calibrated membership values for all conditions and the outcome
- consistency scores for all 15 observed truth table corners
- every term of the parsimonious solution, with consistency and coverage
- every term of the complex solution, with consistency and coverage

Cross-validation against fsQCA 4.x, the Windows and macOS application, has not yet been done.

## All eight notebooks execute end to end

Every notebook in `chNN-*/notebooks/` runs without error from a clean kernel, tested with `nbclient` against the data in this repository. The pip install cell is skipped during automated testing since the environment is already provisioned; it is left in place for Colab users.

## Not yet verified

**Graphical-route procedures have not been re-walked end to end.** The book documents click-by-click routes through GeoDa, the ASU MGWR application, QGIS with the Semi-Automatic Classification Plugin, and fsQCA 4.x. Menu labels move between releases, so these should be checked against whatever version you are running.

**The spreadsheet workbook for Chapter 9 is not in this repository.** See `ch09-ahp-family/WORKBOOK.md` for why, and what stands in for it in the meantime.

## Environment

Results were produced on Python 3.11 with the versions pinned in `requirements.txt`, and R 4.3.3 with QCA 3.25.3 for the fsQCA cross-validation. `torch` is not required anywhere in this repository.

## Reporting a discrepancy

If a published number does not reproduce for you, please open an issue with your Python version, the output of `pip freeze`, and the number you obtained. Confirmed discrepancies will be listed here with credit.
