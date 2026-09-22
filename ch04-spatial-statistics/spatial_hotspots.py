"""
spatial_hotspots.py
-------------------
Global South Urban Methods — Chapter 3 companion module.
Global Moran's I, LISA, and Getis-Ord Gi* in one reusable pipeline.

Usage:
    python spatial_hotspots.py data/phnompenh_communes.geojson informal_rate

Requires: geopandas, libpysal, esda, matplotlib, mapclassify, statsmodels
"""
import sys
import geopandas as gpd
from libpysal import weights
from esda.moran import Moran, Moran_Local
from esda.getisord import G_Local
from statsmodels.stats.multitest import multipletests


def load_and_check(path, value_col, min_n=30):
    gdf = gpd.read_file(path).dropna(subset=[value_col])
    problems = []
    if len(gdf) < min_n:
        problems.append(f"Only {len(gdf)} units (< {min_n}): results unstable.")
    if gdf.crs is None or gdf.crs.is_geographic:
        problems.append("CRS is geographic/undefined; reproject to a metric CRS for distance weights.")
    for p in problems:
        print("  [check]", p)
    return gdf


def build_weights(gdf, kind="queen", k=6):
    if kind == "queen":
        w = weights.Queen.from_dataframe(gdf, use_index=False)
    else:
        w = weights.KNN.from_dataframe(gdf, k=k, use_index=False)
    w.transform = "r"
    if w.islands:
        print(f"  [check] {len(w.islands)} island(s) with no neighbours -> consider KNN.")
    return w


def global_moran(gdf, col, w, perms=999):
    mi = Moran(gdf[col], w, permutations=perms)
    print(f"Global Moran's I = {mi.I:.3f}  (pseudo p = {mi.p_sim:.3f})")
    return mi


def lisa(gdf, col, w, perms=999, fdr=True):
    lm = Moran_Local(gdf[col], w, permutations=perms)
    gdf["lisa_q"] = lm.q               # 1=HH 2=LH 3=LL 4=HL
    sig = lm.p_sim < 0.05
    if fdr:
        sig, _, _, _ = multipletests(lm.p_sim, alpha=0.05, method="fdr_bh")
    gdf["lisa_sig"] = sig
    labels = {1: "High-High", 2: "Low-High", 3: "Low-Low", 4: "High-Low"}
    gdf["lisa_label"] = [labels[q] if s else "ns"
                         for q, s in zip(gdf["lisa_q"], gdf["lisa_sig"])]
    print(gdf.loc[gdf.lisa_sig, "lisa_label"].value_counts().to_string())
    return gdf


def getis_gi_star(gdf, col, w, perms=999):
    gi = G_Local(gdf[col], w, star=True, permutations=perms)
    gdf["gi_z"] = gi.Zs
    def band(z):
        if z > 2.58: return "Hot 99%"
        if z > 1.96: return "Hot 95%"
        if z < -2.58: return "Cold 99%"
        if z < -1.96: return "Cold 95%"
        return "ns"
    gdf["gi_band"] = [band(z) for z in gdf["gi_z"]]
    print(gdf["gi_band"].value_counts().to_string())
    return gdf


def run(path, col):
    gdf = load_and_check(path, col)
    w = build_weights(gdf, "queen")
    global_moran(gdf, col, w)
    gdf = lisa(gdf, col, w)
    gdf = getis_gi_star(gdf, col, w)
    out = path.replace(".geojson", "_results.geojson")
    gdf.to_file(out, driver="GeoJSON")
    print("saved ->", out)
    return gdf


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "data/phnompenh_communes.geojson"
    c = sys.argv[2] if len(sys.argv) > 2 else "informal_rate"
    run(p, c)
