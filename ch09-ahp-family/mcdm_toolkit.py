"""
mcdm_toolkit.py
---------------
Global South Urban Methods — Chapter 8 companion module.
AHP, group AHP, fuzzy-AHP (Buckley geometric mean), ANP, and IPA.

Written so every step can be checked by hand against §8.4.

    python mcdm_toolkit.py
"""
import numpy as np
import pandas as pd

# Saaty's random consistency index (Saaty 1980)
RI = {1: 0.00, 2: 0.00, 3: 0.58, 4: 0.90, 5: 1.12, 6: 1.24,
      7: 1.32, 8: 1.41, 9: 1.45, 10: 1.49, 11: 1.51, 12: 1.48}


# ================================================================== 1. AHP
def priority_vector(A, method="eigen"):
    """Derive weights from a pairwise comparison matrix.

    method='eigen' : principal eigenvector (Saaty's original)
    method='gmean' : row geometric mean (simpler, nearly identical, and the
                     only method that survives being written as a spreadsheet
                     formula -- see the workbook in this folder)
    """
    A = np.asarray(A, float)
    if method == "gmean":
        w = np.prod(A, axis=1) ** (1 / A.shape[0])
    else:
        vals, vecs = np.linalg.eig(A)
        w = np.real(vecs[:, np.argmax(np.real(vals))])
        w = np.abs(w)
    return w / w.sum()


def consistency(A, w=None):
    """Consistency ratio. CR < 0.10 is the conventional acceptance threshold."""
    A = np.asarray(A, float)
    n = A.shape[0]
    if w is None:
        w = priority_vector(A)
    lam_max = float((A @ w / w).mean())
    ci = (lam_max - n) / (n - 1) if n > 1 else 0.0
    ri = RI.get(n, 1.49)
    cr = ci / ri if ri else 0.0
    return {"lambda_max": round(lam_max, 4), "CI": round(ci, 4),
            "RI": ri, "CR": round(cr, 4), "acceptable": bool(cr < 0.10)}


def ahp(A, labels=None, method="eigen"):
    w = priority_vector(A, method)
    out = consistency(A, w)
    out["weights"] = dict(zip(labels or range(len(w)), np.round(w, 4)))
    return out


# ======================================================= 2. GROUP AGGREGATION
def aggregate_judgments(matrices):
    """AIJ — aggregate individual judgments by geometric mean.

    The geometric mean is used rather than the arithmetic mean because it is
    the only aggregation that preserves reciprocity: if the group's a_ij is
    the geometric mean of individual a_ij, then a_ji is automatically 1/a_ij.
    """
    M = np.asarray(matrices, float)
    return np.exp(np.log(M).mean(axis=0))


def screen_respondents(matrices, cr_max=0.10):
    """Report each respondent's CR so inconsistent ones can be excluded."""
    rows = []
    for i, A in enumerate(matrices):
        c = consistency(A)
        rows.append({"respondent": i, "CR": c["CR"], "keep": c["acceptable"]})
    return pd.DataFrame(rows)


# ============================================================ 3. FUZZY AHP
# Triangular fuzzy numbers for the Saaty scale: (lower, modal, upper)
TFN = {1: (1, 1, 1), 2: (1, 2, 3), 3: (2, 3, 4), 4: (3, 4, 5), 5: (4, 5, 6),
       6: (5, 6, 7), 7: (6, 7, 8), 8: (7, 8, 9), 9: (8, 9, 9)}


def to_fuzzy(A):
    """Crisp Saaty matrix -> triangular fuzzy matrix, shape (n, n, 3)."""
    A = np.asarray(A, float)
    n = A.shape[0]
    F = np.zeros((n, n, 3))
    for i in range(n):
        for j in range(n):
            v = A[i, j]
            if v >= 1:
                F[i, j] = TFN.get(int(round(v)), (v, v, v))
            else:                                   # reciprocal: invert & swap
                l, m, u = TFN.get(int(round(1 / v)), (1/v, 1/v, 1/v))
                F[i, j] = (1 / u, 1 / m, 1 / l)
    return F


def fuzzy_ahp(A, labels=None):
    """Buckley's geometric-mean method.

    Chang's extent analysis is more widely cited but can assign a weight of
    exactly zero to a criterion that is plainly relevant (Wang, Luo & Hua,
    2008). Buckley's method has no such failure mode, so it is used here.
    """
    F = to_fuzzy(A)
    n = F.shape[0]
    r = np.prod(F, axis=1) ** (1 / n)              # fuzzy geometric mean per row
    r_sum = r.sum(axis=0)
    w = np.empty_like(r)
    w[:, 0] = r[:, 0] / r_sum[2]                   # l / sum(u)
    w[:, 1] = r[:, 1] / r_sum[1]
    w[:, 2] = r[:, 2] / r_sum[0]                   # u / sum(l)
    crisp = w.mean(axis=1)                         # centroid defuzzification
    crisp = crisp / crisp.sum()
    lab = list(labels or range(n))
    return {"fuzzy_weights": {k: tuple(np.round(v, 4)) for k, v in zip(lab, w)},
            "weights": dict(zip(lab, np.round(crisp, 4)))}


# ================================================================== 4. ANP
def limit_matrix(supermatrix, max_iter=200, tol=1e-8):
    """Raise a column-stochastic supermatrix to its limit (ANP synthesis)."""
    W = np.asarray(supermatrix, float)
    W = W / W.sum(axis=0, keepdims=True)
    prev = W.copy()
    for k in range(max_iter):
        cur = prev @ W
        cur = cur / cur.sum(axis=0, keepdims=True)
        if np.abs(cur - prev).max() < tol:
            return cur, k + 1
        prev = cur
    return prev, max_iter


def anp(supermatrix, labels=None):
    L, it = limit_matrix(supermatrix)
    w = L[:, 0] / L[:, 0].sum()
    return {"weights": dict(zip(labels or range(len(w)), np.round(w, 4))),
            "iterations": it}


# ================================================================== 5. IPA
def ipa(importance, performance, labels):
    """Importance-Performance Analysis with mean-based cross-hairs."""
    imp, perf = np.asarray(importance, float), np.asarray(performance, float)
    mi, mp = imp.mean(), perf.mean()
    quad = []
    for i, p in zip(imp, perf):
        if i >= mi and p < mp:   quad.append("Concentrate here")
        elif i >= mi and p >= mp: quad.append("Keep up the good work")
        elif i < mi and p < mp:   quad.append("Low priority")
        else:                     quad.append("Possible overkill")
    return pd.DataFrame({"item": labels, "importance": imp.round(4),
                         "performance": perf.round(2), "quadrant": quad}), (mi, mp)


if __name__ == "__main__":
    labels = ["water", "sanitation", "roads", "electricity", "tenure"]
    A = np.array([[1, 2, 3, 4, 1/2],
                  [1/2, 1, 2, 3, 1/3],
                  [1/3, 1/2, 1, 2, 1/4],
                  [1/4, 1/3, 1/2, 1, 1/5],
                  [2, 3, 4, 5, 1]])
    r = ahp(A, labels)
    print("AHP weights:", r["weights"])
    print("CR =", r["CR"], "acceptable:", r["acceptable"])
    print("Fuzzy AHP:", fuzzy_ahp(A, labels)["weights"])
