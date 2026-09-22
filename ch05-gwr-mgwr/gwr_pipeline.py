"""
gwr_pipeline.py
---------------
Global South Urban Methods — Chapter 4 companion module.
OLS -> GWR -> MGWR comparison in one reusable pipeline.

Usage:
    python gwr_pipeline.py data/phnompenh_parcels.geojson land_price dist_cbd road_access flood_risk

Requires: geopandas, mgwr, spglm, numpy, matplotlib
"""
import sys
import numpy as np
import geopandas as gpd
from mgwr.gwr import GWR, MGWR
from mgwr.sel_bw import Sel_BW
from spglm.family import Gaussian


def load(path, y_col, x_cols):
    gdf = gpd.read_file(path).dropna(subset=[y_col] + list(x_cols))
    if len(gdf) < 100:
        print(f"  [check] only {len(gdf)} observations; GWR wants 100+ for stable local estimates.")
    if gdf.crs is None or gdf.crs.is_geographic:
        print("  [check] CRS is geographic; reproject to a metric CRS (bandwidth is in map units).")
    return gdf


def prepare(gdf, y_col, x_cols, standardize=True):
    y = gdf[y_col].values.reshape(-1, 1)
    X = gdf[list(x_cols)].values
    coords = list(zip(gdf.geometry.centroid.x, gdf.geometry.centroid.y))
    if standardize:                       # required for MGWR bandwidth comparability
        X = (X - X.mean(axis=0)) / X.std(axis=0)
        y = (y - y.mean()) / y.std()
    return y, X, coords


def run_ols(y, X):
    Xc = np.hstack([np.ones((X.shape[0], 1)), X])
    beta = np.linalg.lstsq(Xc, y, rcond=None)[0]
    resid = y - Xc @ beta
    r2 = 1 - (resid**2).sum() / ((y - y.mean())**2).sum()
    print(f"OLS  R2 = {r2:.3f}")
    return beta, r2


def run_gwr(y, X, coords, fixed=False):
    bw = Sel_BW(coords, y, X, fixed=fixed).search()
    model = GWR(coords, y, X, bw, fixed=fixed).fit()
    print(f"GWR  bandwidth = {bw}   AICc = {model.aicc:.1f}   R2 = {model.R2:.3f}")
    return model, bw


def run_mgwr(y, X, coords):
    selector = Sel_BW(coords, y, X, multi=True)
    bws = selector.search(multi_bw_min=[2])
    model = MGWR(coords, y, X, selector).fit()
    print(f"MGWR bandwidths = {list(bws)}   AICc = {model.aicc:.1f}   R2 = {model.R2:.3f}")
    return model, bws


def run(path, y_col, x_cols):
    gdf = load(path, y_col, x_cols)
    y, X, coords = prepare(gdf, y_col, x_cols)
    run_ols(y, X)
    gwr_model, bw = run_gwr(y, X, coords)
    mgwr_model, bws = run_mgwr(y, X, coords)

    # attach local coefficients (GWR) for mapping
    for i, name in enumerate(["intercept"] + list(x_cols)):
        gdf[f"gwr_b_{name}"] = gwr_model.params[:, i]
    out = path.replace(".geojson", "_results.geojson")
    gdf.to_file(out, driver="GeoJSON")
    print("saved ->", out)
    return gdf, gwr_model, mgwr_model


if __name__ == "__main__":
    p = sys.argv[1] if len(sys.argv) > 1 else "data/phnompenh_parcels.geojson"
    y = sys.argv[2] if len(sys.argv) > 2 else "land_price"
    xs = sys.argv[3:] if len(sys.argv) > 3 else ["dist_cbd", "road_access", "flood_risk"]
    run(p, y, xs)
