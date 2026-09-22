"""
vgi_toolkit.py
--------------
Global South Urban Methods — Chapter 7 companion module.
Volunteered geographic information (OSM) for urban analysis, with the
completeness/bias assessment that must precede any substantive use.

    python vgi_toolkit.py

Requires: geopandas, numpy, pandas, networkx, shapely
"""
import numpy as np
import pandas as pd
import geopandas as gpd


# ============================================================== 1. COMPLETENESS
def intrinsic_quality(buildings, zones, zone_col="zone_id"):
    """Quality indicators computable from OSM alone — no reference data needed.

    Use when you have nothing to compare against. These do not measure
    completeness directly; they measure *how much attention* an area received.
    """
    g = buildings.groupby(zone_col)
    out = pd.DataFrame({
        "n_features":      g.size(),
        "mean_contributors": g["n_contributors"].mean().round(2),
        "median_last_edit":  g["last_edit_year"].median(),
        "attribute_completeness": g["building_tag"].apply(
            lambda s: round((s != "yes").mean(), 3)),      # share with a real tag
    }).reset_index()
    return zones[[zone_col]].merge(out, on=zone_col, how="left").fillna(0)


def extrinsic_completeness(buildings, reference, zones, zone_col="zone_id",
                           ref_col="buildings_true"):
    """Compare OSM counts against an independent reference (census, GHSL,
    imagery-derived counts from Chapter 6). This is the defensible method."""
    obs = buildings.groupby(zone_col).size().rename("n_osm")
    df = (zones[[zone_col]].merge(obs, on=zone_col, how="left").fillna(0)
          .merge(reference[[zone_col, ref_col]], on=zone_col, how="left"))
    df["completeness"] = (df["n_osm"] / df[ref_col]).round(3)
    return df


def saturation_curve(buildings, zone_col="zone_id", year_col="last_edit_year"):
    """Cumulative mapped features per year. A curve that has flattened
    suggests the area is近 complete; one still rising suggests it is not."""
    c = (buildings.groupby([zone_col, year_col]).size()
         .groupby(level=0).cumsum().rename("cumulative").reset_index())
    return c


# ========================================================= 2. STREET NETWORKS
def orientation_entropy(roads, n_bins=36):
    """Boeing-style orientation entropy: how ordered is the street grid?

    ~1.0  -> perfectly disordered (organic/informal fabric)
    lower -> ordered grid (planned fabric)
    """
    angs = []
    for geom in roads.geometry:
        c = np.asarray(geom.coords)
        d = np.diff(c, axis=0)
        a = np.degrees(np.arctan2(d[:, 1], d[:, 0])) % 180     # undirected
        angs.extend(a.tolist())
    if not angs:
        return np.nan
    hist, _ = np.histogram(angs, bins=n_bins, range=(0, 180))
    p = hist / hist.sum()
    p = p[p > 0]
    return round(float(-(p * np.log(p)).sum() / np.log(n_bins)), 3)


def network_metrics(roads, zones, zone_col="zone_id"):
    """Per-zone street density and orientation entropy."""
    rows = []
    for _, z in zones.iterrows():
        sub = roads[roads[zone_col] == z[zone_col]]
        area_km2 = z.geometry.area / 1e6
        length_km = sub.geometry.length.sum() / 1000
        rows.append({zone_col: z[zone_col],
                     "street_km_per_km2": round(length_km / area_km2, 2),
                     "orientation_entropy": orientation_entropy(sub)})
    return pd.DataFrame(rows)


# ============================================================ 3. ACCESSIBILITY
def nearest_facility(zones, pois, amenity=None, zone_col="zone_id"):
    """Distance from each zone centroid to the nearest facility (metres)."""
    p = pois if amenity is None else pois[pois["amenity"] == amenity]
    if len(p) == 0:
        return pd.DataFrame({zone_col: zones[zone_col], "dist_m": np.nan})
    cent = zones.geometry.centroid
    coords = np.array([(g.x, g.y) for g in p.geometry])
    rows = []
    for zi, c in zip(zones[zone_col], cent):
        d = np.hypot(coords[:, 0] - c.x, coords[:, 1] - c.y).min()
        rows.append({zone_col: zi, "dist_m": round(float(d))})
    return pd.DataFrame(rows)


# ======================================================== 4. BIAS CORRECTION
def adjust_for_completeness(values, completeness, floor=0.2):
    """Scale an OSM-derived count by estimated completeness.

    Only defensible where completeness was estimated against a reference.
    Values below `floor` completeness are returned as NaN — the correction
    factor becomes too large to trust.
    """
    c = np.asarray(completeness, float)
    v = np.asarray(values, float)
    out = np.where(c >= floor, v / np.maximum(c, 1e-6), np.nan)
    return np.round(out, 1)


if __name__ == "__main__":
    b = gpd.read_file("data/osm_buildings.geojson")
    r = gpd.read_file("data/osm_roads.geojson")
    z = gpd.read_file("data/zones.geojson")
    t = pd.read_csv("data/ground_truth.csv")

    comp = extrinsic_completeness(b, t, z)
    print("completeness by zone type:")
    print(comp.merge(t[["zone_id", "zone_type"]], on="zone_id")
              .groupby("zone_type")["completeness"].mean().round(3).to_string())
    print("\norientation entropy (all roads):", orientation_entropy(r))
