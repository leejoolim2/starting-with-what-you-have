"""
fsqca.py
--------
Global South Urban Methods — Chapter 5 companion module.
A transparent, dependency-light fuzzy-set QCA implementation:
calibration, set operations, necessity, truth table, sufficiency,
and Boolean minimisation (complex/conservative solution).

Written to be *readable* rather than fast — every step mirrors the
procedure described in §5.4 so you can check the arithmetic by hand.

Usage:
    import fsqca
    fs = fsqca.calibrate(df["tenure"], full_in=8, crossover=5, full_out=2)
"""
import itertools
import numpy as np
import pandas as pd


# ---------------------------------------------------------------- calibration
def calibrate(x, full_in, crossover, full_out):
    """Direct method of calibration (Ragin 2008), log-odds based.

    full_in   : raw value = full membership   (fuzzy 0.95)
    crossover : raw value = maximum ambiguity (fuzzy 0.50)
    full_out  : raw value = full non-membership (fuzzy 0.05)
    """
    x = np.asarray(x, dtype=float)
    dev = x - crossover
    # separate scaling above and below the crossover point
    up = np.log(0.95 / 0.05) / (full_in - crossover)
    dn = np.log(0.95 / 0.05) / (crossover - full_out)
    scalar = np.where(dev >= 0, dev * up, dev * dn)
    fuzzy = np.exp(scalar) / (1 + np.exp(scalar))
    return np.round(np.clip(fuzzy, 0.001, 0.999), 3)


# ------------------------------------------------------------- set operations
def NOT(a):
    return 1 - np.asarray(a, dtype=float)


def AND(*sets):
    return np.min(np.vstack([np.asarray(s, dtype=float) for s in sets]), axis=0)


def OR(*sets):
    return np.max(np.vstack([np.asarray(s, dtype=float) for s in sets]), axis=0)


# --------------------------------------------------- consistency and coverage
def consistency(X, Y):
    """Sufficiency consistency: sum(min(X,Y)) / sum(X)."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    return float(np.sum(np.minimum(X, Y)) / np.sum(X))


def coverage(X, Y):
    """Sufficiency coverage: sum(min(X,Y)) / sum(Y)."""
    X, Y = np.asarray(X, float), np.asarray(Y, float)
    return float(np.sum(np.minimum(X, Y)) / np.sum(Y))


def nec_consistency(X, Y):
    """Necessity consistency: sum(min(X,Y)) / sum(Y)."""
    return coverage(X, Y)


def nec_coverage(X, Y):
    """Necessity coverage (relevance): sum(min(X,Y)) / sum(X)."""
    return consistency(X, Y)


def necessity_table(conditions: dict, outcome, include_negations=True):
    """Screen every condition (and its negation) for necessity."""
    rows = []
    items = list(conditions.items())
    if include_negations:
        items += [("~" + k, NOT(v)) for k, v in conditions.items()]
    for name, cond in items:
        rows.append({
            "condition": name,
            "consistency": round(nec_consistency(cond, outcome), 3),
            "coverage": round(nec_coverage(cond, outcome), 3),
        })
    return (pd.DataFrame(rows)
            .sort_values("consistency", ascending=False)
            .reset_index(drop=True))


# ----------------------------------------------------------------- truth table
def truth_table(conditions: dict, outcome, freq_cutoff=1, cons_cutoff=0.80):
    """Build the fuzzy-set truth table.

    Each case belongs to exactly one corner of the vector space (the corner
    where its membership exceeds 0.5 in every dimension).
    """
    names = list(conditions)
    mat = np.vstack([np.asarray(conditions[n], float) for n in names]).T
    outcome = np.asarray(outcome, float)
    k = len(names)

    rows = []
    for corner in itertools.product([0, 1], repeat=k):
        # membership in this corner = min over conditions of (x or 1-x)
        memb = np.min(np.where(np.array(corner) == 1, mat, 1 - mat), axis=1)
        cases = np.where(memb > 0.5)[0]      # cases *assigned* to this corner
        n = len(cases)
        # Consistency is computed over ALL cases weighted by their membership in
        # this corner, not only over the cases assigned to it. Restricting to
        # assigned cases is a common implementation error and produces values
        # that disagree with the reference software.
        if n == 0:
            cons = np.nan
        else:
            cons = consistency(memb, outcome)
        rows.append({
            **{names[i]: corner[i] for i in range(k)},
            "n": n,
            "consistency": np.nan if np.isnan(cons) else round(cons, 3),
        })

    tt = pd.DataFrame(rows)
    codes = []
    for _, r in tt.iterrows():
        if r["n"] < freq_cutoff or pd.isna(r["consistency"]):
            codes.append("?")                       # logical remainder
        elif r["consistency"] >= cons_cutoff:
            codes.append(1)
        else:
            codes.append(0)
    tt["outcome"] = pd.Series(codes, dtype=object)
    return tt


# ------------------------------------------------- Boolean minimisation (QM)
def _combine(a, b):
    """Combine two implicants differing in exactly one literal -> '-'."""
    diff = [i for i, (x, y) in enumerate(zip(a, b)) if x != y]
    if len(diff) != 1:
        return None
    out = list(a)
    out[diff[0]] = "-"
    return tuple(out)


def minimise(tt, names, outcome_col="outcome", use_remainders=False):
    """Boolean minimisation on the rows coded 1.

    use_remainders=False -> complex (conservative) solution: logical
        remainders are never used as counterfactuals.
    use_remainders=True  -> parsimonious solution: remainders may be used
        freely as 'don't care' terms, but need not be covered.

    Selection is deterministic (ties broken by sorted order) so repeated
    runs return identical solutions.
    """
    terms = {tuple(str(r[n]) for n in names)
             for _, r in tt.iterrows() if r[outcome_col] == 1}
    if not terms:
        return []

    pool = set(terms)
    if use_remainders:
        pool |= {tuple(str(r[n]) for n in names)
                 for _, r in tt.iterrows() if r[outcome_col] == "?"}

    primes, current = set(), set(pool)
    while current:
        nxt, used = set(), set()
        for a, b in itertools.combinations(sorted(current), 2):
            c = _combine(a, b)
            if c:
                nxt.add(c)
                used.add(a)
                used.add(b)
        primes |= (current - used)
        current = nxt

    def covers(prime, term):
        return all(p == "-" or p == t for p, t in zip(prime, term))

    # only the observed positive terms must be covered
    uncovered, solution = set(terms), []
    while uncovered:
        # deterministic: most coverage first, then fewest literals, then sorted
        best = min(sorted(primes),
                   key=lambda p: (-sum(covers(p, t) for t in uncovered),
                                  sum(1 for v in p if v != "-"),
                                  p))
        solution.append(best)
        uncovered -= {t for t in uncovered if covers(best, t)}
        primes.discard(best)
    return sorted(solution)


def express(solution, names):
    """Render implicants as readable Boolean expressions."""
    out = []
    for imp in solution:
        parts = [(n if v == "1" else "~" + n)
                 for n, v in zip(names, imp) if v != "-"]
        out.append(" * ".join(parts))
    return out


def solution_table(solution, names, conditions, outcome):
    """Raw/unique coverage and consistency for each path, plus the solution."""
    paths = express(solution, names)
    sets_ = []
    for imp in solution:
        comps = [conditions[n] if v == "1" else NOT(conditions[n])
                 for n, v in zip(names, imp) if v != "-"]
        sets_.append(AND(*comps) if len(comps) > 1 else np.asarray(comps[0]))

    rows = []
    for i, (p, s) in enumerate(zip(paths, sets_)):
        others = [t for j, t in enumerate(sets_) if j != i]
        uniq = (coverage(s, outcome) - coverage(AND(s, OR(*others)), outcome)
                if others else coverage(s, outcome))
        rows.append({"path": p,
                     "consistency": round(consistency(s, outcome), 3),
                     "raw_coverage": round(coverage(s, outcome), 3),
                     "unique_coverage": round(max(uniq, 0), 3)})
    df = pd.DataFrame(rows)
    total = OR(*sets_) if len(sets_) > 1 else sets_[0]
    return df, {"solution_consistency": round(consistency(total, outcome), 3),
                "solution_coverage": round(coverage(total, outcome), 3)}
