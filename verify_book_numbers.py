# -*- coding: utf-8 -*-
"""
verify_book_numbers.py

E1 verification. Runs each chapter's analysis from the repository and compares
the result against the number printed in the book. Any mismatch is a defect in
the book, not in the reader's environment, and is reported as such.

    python verify_book_numbers.py
"""
import warnings; warnings.filterwarnings("ignore")
import sys, os, numpy as np, pandas as pd

R = "/home/claude/repo/starting-with-what-you-have"
results = []


def check(chapter, label, got, expected, tol=0.002):
    ok = abs(float(got) - float(expected)) <= tol
    results.append((chapter, label, float(got), float(expected), ok))
    return ok


# ---------------------------------------------------------------- Chapter 4
def ch04():
    import geopandas as gpd
    from libpysal import weights
    from esda.moran import Moran
    g = gpd.read_file(f"{R}/ch04-spatial-statistics/data/phnompenh_communes_results.geojson")
    w = weights.Queen.from_dataframe(g, use_index=False); w.transform = "r"
    mi = Moran(g["informal_rate"].values, w, permutations=999)
    check("4", "Global Moran's I", mi.I, 0.706, tol=0.02)
    check("4", "High-High units", (g.lisa_label == "High-High").sum(), 18, tol=0)
    check("4", "Gi* hotspots (z>1.96)", (g.gi_z > 1.96).sum(), 6, tol=0)


# ---------------------------------------------------------------- Chapter 5
def ch05():
    import geopandas as gpd
    g = gpd.read_file(f"{R}/ch05-gwr-mgwr/data/phnompenh_parcels_results.geojson")
    r = np.corrcoef(g.true_b_dist, g.gwr_b_dist_cbd)[0, 1]
    check("5", "GWR coefficient recovery r", r, 0.906, tol=0.02)


# ---------------------------------------------------------------- Chapter 6
def ch06():
    sys.path.insert(0, f"{R}/ch06-fsqca")
    import fsqca
    df = pd.read_csv(f"{R}/ch06-fsqca/data/upgrading_cities.csv")
    CONDS = ["tenure", "finance", "participation", "capacity"]
    cond = {c: fsqca.calibrate(df[c], 8, 5, 2) for c in CONDS}
    Y = fsqca.calibrate(df["upgrade_success"], 8, 5, 2)
    tt = fsqca.truth_table(cond, Y, freq_cutoff=1, cons_cutoff=0.80)
    paths = fsqca.minimise(tt, CONDS, use_remainders=True)

    def as_terms(implicant):
        """('-','1','-','1') -> {'finance','capacity'} (present conditions only)"""
        return {CONDS[i] for i, v in enumerate(implicant) if v == "1"}

    found = [as_terms(p) for p in paths]
    check("6", "path finance+capacity recovered",
          1 if {"finance", "capacity"} in found else 0, 1, tol=0)
    # After cross-validation against R's QCA, the parsimonious solution returns
    # `participation` alone, which is more parsimonious than the planted
    # `tenure*participation`. The book documents this.
    check("6", "path participation recovered",
          1 if {"participation"} in found else 0, 1, tol=0)
    check("6", "solution term count (parsimonious)", len(found), 3, tol=0)


# ---------------------------------------------------------------- Chapter 7
def ch07():
    sys.path.insert(0, f"{R}/ch07-remote-sensing")
    import rs_toolkit as rs

    def ndvi(s):
        return (s["nir"] - s["red"]) / (s["nir"] + s["red"] + 1e-9)

    s15, _ = rs.read_scene(f"{R}/ch07-remote-sensing/data/scene_2015.tif")
    s23, _ = rs.read_scene(f"{R}/ch07-remote-sensing/data/scene_2023.tif")
    n15, n23 = ndvi(s15), ndvi(s23)
    check("7", "NDVI 2015 mean", float(np.nanmean(n15)), 0.544, tol=0.01)
    check("7", "NDVI 2023 mean", float(np.nanmean(n23)), 0.512, tol=0.01)
    check("7", "NDVI to LST correlation",
          np.corrcoef(n23.ravel(), s23["lst_c"].ravel())[0, 1], -0.454, tol=0.02)
    check("7", "SUHI-style warming 2015 to 2023",
          float(np.nanmean(s23["lst_c"]) - np.nanmean(s15["lst_c"])), 1.64, tol=0.05)


# ---------------------------------------------------------------- Chapter 8
def ch08():
    import geopandas as gpd
    sys.path.insert(0, f"{R}/ch08-vgi-osm")
    b = gpd.read_file(f"{R}/ch08-vgi-osm/data/osm_buildings.geojson")
    gt = pd.read_csv(f"{R}/ch08-vgi-osm/data/ground_truth.csv")
    obs = b.groupby("zone_id").size().rename("n_osm")
    d = gt.set_index("zone_id").join(obs).fillna(0)
    d["completeness"] = d.n_osm / d.buildings_true
    m = d.groupby("zone_type").completeness.mean().sort_values(ascending=False)
    vals = m.values  # highest = central, middle = formal periphery, lowest = informal
    check("8", "completeness, central", vals[0], 0.949, tol=0.02)
    check("8", "completeness, formal periphery", vals[1], 0.698, tol=0.02)
    check("8", "completeness, informal", vals[2], 0.349, tol=0.02)
    contrib = b.groupby("zone_id").n_contributors.mean().reindex(d.index)
    check("8", "contributor count vs completeness r",
          np.corrcoef(contrib, d.completeness)[0, 1], 0.959, tol=0.03)


# ---------------------------------------------------------------- Chapter 9
def ch09():
    sys.path.insert(0, f"{R}/ch09-ahp-family")
    import mcdm_toolkit as m
    mats = np.load(f"{R}/ch09-ahp-family/data/matrices.npy")
    L = ["water", "sanitation", "roads", "electricity", "tenure"]
    scr = m.screen_respondents(mats, cr_max=0.10)
    check("9", "respondents dropped", int((~scr.keep).sum()), 5, tol=0)
    wk = m.ahp(m.aggregate_judgments(mats[scr.keep.values]), L)["weights"]
    wa = m.ahp(m.aggregate_judgments(mats), L)["weights"]
    check("9", "tenure weight, screened", wk["tenure"], 0.4386, tol=0.001)
    check("9", "tenure weight, unscreened", wa["tenure"], 0.3847, tol=0.001)
    check("9", "screening effect (pp)", (wk["tenure"] - wa["tenure"]) * 100, 5.4, tol=0.15)


# --------------------------------------------------------------- Chapter 10
def ch10():
    sys.path.insert(0, f"{R}/ch10-spatial-ml-xai")
    import spatial_ml as sm
    df = pd.read_csv(f"{R}/ch10-spatial-ml-xai/data/urban_heat.csv")
    F = ["ndvi", "bld_density", "dist_water_m", "impervious", "elevation_m"]
    X, y, co = df[F].values, df.lst_c.values, df[["x", "y"]].values
    mo = sm.fit_models(X, y)
    for name, exp_r, exp_s in [("OLS", 0.726, 0.476), ("RandomForest", 0.931, 0.834)]:
        r = sm.random_cv(mo[name], X, y)["r2_mean"]
        s = sm.spatial_block_cv(mo[name], X, y, co)["r2_mean"]
        check("10", f"{name} random CV", r, exp_r, tol=0.02)
        check("10", f"{name} spatial block CV", s, exp_s, tol=0.02)
    rf = mo["RandomForest"]
    xs, pdv = sm.partial_dependence(rf, X, 0)
    slope = np.gradient(pdv, xs)
    thr = xs[np.argmax(slope > slope.min() * 0.25)]
    check("10", "recovered NDVI threshold", thr, 0.450, tol=0.02)
    sv, _ = sm.shap_values(rf, X, F)
    j = F.index("bld_density")
    d_core = np.hypot(co[:, 0] - 6000, co[:, 1] - 6000)
    w_core = np.exp(-(d_core ** 2) / (2 * 3000 ** 2))
    check("10", "local SHAP vs planted effect r",
          np.corrcoef(sv[:, j], w_core * df.bld_density.values)[0, 1], 0.981, tol=0.02)


# --------------------------------------------------------------- Chapter 11
def ch11():
    sys.path.insert(0, f"{R}/ch11-text-and-image")
    import text_toolkit as tt, image_toolkit as it
    en = pd.read_csv(f"{R}/ch11-text-and-image/data/complaints_en.csv")
    lr = pd.read_csv(f"{R}/ch11-text-and-image/data/complaints_lowres.csv")
    check("11", "English tokens per type",
          tt.vocab_profile(en.text)["tokens_per_type"], 442.72, tol=1.0)
    check("11", "low-resource tokens per type",
          tt.vocab_profile(lr.text)["tokens_per_type"], 84.04, tol=1.0)
    check("11", "English F1 (word n-grams)",
          tt.evaluate(en.text, en.label)["f1_macro_mean"], 0.919, tol=0.01)
    check("11", "low-resource F1 (word n-grams)",
          tt.evaluate(lr.text, lr.label)["f1_macro_mean"], 0.827, tol=0.015)
    _, W, _ = tt.topic_model(en.text.values, n_topics=5)
    ct = tt.topic_label_crosstab(W, en.label.values)
    check("11", "topic-label agreement",
          ct.values.max(axis=1).sum() / ct.values.sum(), 0.929, tol=0.02)
    cwd = os.getcwd(); os.chdir(f"{R}/ch11-text-and-image")
    X, y, dfi = it.build_feature_table("data/streetview_labels.csv")
    gi = it.FEATURE_NAMES.index("GVI")
    check("11", "GVI vs planted green r",
          np.corrcoef(dfi.true_green, X[:, gi])[0, 1], 1.0000, tol=0.001)
    check("11", "F1, GVI alone", it.evaluate(X[:, [gi]], y)["f1_macro_mean"], 0.696, tol=0.02)
    check("11", "F1, all features", it.evaluate(X, y)["f1_macro_mean"], 0.900, tol=0.02)
    os.chdir(cwd)


if __name__ == "__main__":
    for fn in (ch04, ch05, ch06, ch07, ch08, ch09, ch10, ch11):
        try:
            fn()
        except Exception as e:
            results.append((fn.__name__[2:], f"ERROR: {type(e).__name__}: {e}", 0, 0, False))

    print(f"{'Ch':>3}  {'Quantity':<42} {'Reproduced':>12} {'In book':>10}  ")
    print("-" * 78)
    npass = 0
    for ch, label, got, exp, ok in results:
        mark = "PASS" if ok else "FAIL"
        npass += ok
        print(f"{ch:>3}  {label:<42} {got:>12.4f} {exp:>10.4f}  {mark}")
    print("-" * 78)
    print(f"{npass}/{len(results)} checks passed")
