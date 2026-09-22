# The spreadsheet workbook (pending translation)

Chapter 9 describes a spreadsheet workbook that performs the whole AHP procedure with live formulas, so that the method can be followed without writing any code. Every calculated cell is a real formula rather than a pasted value, which means a reader can trace the arithmetic from pairwise judgements through the eigenvector to the final weights.

**The workbook and its generator are not in this repository yet.** They exist for the Korean edition, where all sheet labels, guidance text, and cell annotations are in Korean. Shipping a partially translated version would be worse than shipping none, because a workbook whose labels disagree with the book's terminology is actively misleading.

The English workbook will be added here once its labels are translated and its formulas re-verified against the Python implementation. The verification standard is the one the Korean edition already met: 197 live formulas checked cell by cell against `mcdm_toolkit.py` and agreeing to five decimal places.

Until then, `mcdm_toolkit.py` in this directory performs the same calculations, and section 9.4-C of the book documents the route through the free fsQCA-style graphical tools.
