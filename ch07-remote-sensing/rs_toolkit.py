"""
rs_toolkit.py
-------------
Global South Urban Methods — Chapter 6 companion module.
Landsat Collection-2 Level-2 processing: scaling, spectral indices,
land surface temperature, supervised classification, change detection,
and surface urban heat island (SUHI) statistics.

Works on any Landsat 8/9 C2 L2 GeoTIFF stacked as
    SR_B2 SR_B3 SR_B4 SR_B5 SR_B6 SR_B7 ST_B10
Usage:
    python rs_toolkit.py data/scene_2015.tif data/scene_2023.tif
"""
import sys
import numpy as np
import rasterio

# ---- official Landsat Collection 2 Level-2 scaling -------------------------
SR_SCALE, SR_OFFSET = 0.0000275, -0.2      # surface reflectance -> 0-1
ST_SCALE, ST_OFFSET = 0.00341802, 149.0    # surface temperature -> Kelvin
BANDS = {"blue": 0, "green": 1, "red": 2, "nir": 3, "swir1": 4, "swir2": 5}


def read_scene(path):
    """Read a stacked scene and apply the official scale factors."""
    with rasterio.open(path) as src:
        arr = src.read().astype(np.float32)
        prof = src.profile
    sr = arr[:6] * SR_SCALE + SR_OFFSET          # reflectance
    lst_k = arr[6] * ST_SCALE + ST_OFFSET        # Kelvin
    return {k: sr[i] for k, i in BANDS.items()} | {"lst_c": lst_k - 273.15}, prof


# ---- spectral indices ------------------------------------------------------
def _norm(a, b):
    return (a - b) / (a + b + 1e-10)


def ndvi(s):  return _norm(s["nir"],   s["red"])     # vegetation
def ndbi(s):  return _norm(s["swir1"], s["nir"])     # built-up
def ndwi(s):  return _norm(s["green"], s["nir"])     # water
def nbr(s):   return _norm(s["nir"],   s["swir2"])   # burn / bare


# ---- land surface temperature from Level-1 (teaching path) ----------------
K1_B10, K2_B10 = 774.8853, 1321.0789     # Landsat 8 TIRS band 10 constants
LAMBDA_B10 = 10.895e-6                   # m
RHO = 1.438e-2                           # m K  (h*c/sigma)


def brightness_temperature(radiance):
    """Convert TOA spectral radiance to at-sensor brightness temperature (K)."""
    return K2_B10 / np.log((K1_B10 / radiance) + 1.0)


def emissivity_from_ndvi(nd, ndvi_soil=0.2, ndvi_veg=0.5):
    """NDVI-threshold method: fractional vegetation cover -> emissivity."""
    pv = np.clip((nd - ndvi_soil) / (ndvi_veg - ndvi_soil), 0, 1) ** 2
    return 0.004 * pv + 0.986


def lst_single_channel(bt_kelvin, emis):
    """Single-channel (mono-window) correction of brightness temperature."""
    return bt_kelvin / (1 + (LAMBDA_B10 * bt_kelvin / RHO) * np.log(emis)) - 273.15


# ---- supervised classification --------------------------------------------
def build_features(s):
    """Stack reflectance bands + indices into an (n_pixels, n_features) matrix."""
    feats = [s[b] for b in BANDS] + [ndvi(s), ndbi(s), ndwi(s), s["lst_c"]]
    return np.stack(feats, -1).reshape(-1, len(feats))


def classify(scene, labels, n_train=200, seed=0):
    """Random-forest supervised classification with a train/test split."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, cohen_kappa_score

    X, y = build_features(scene), labels.ravel()
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(y == c)[0],
                                     min(n_train, (y == c).sum()), replace=False)
                          for c in np.unique(y)])
    Xtr, Xte, ytr, yte = train_test_split(X[idx], y[idx], test_size=.3,
                                          random_state=seed, stratify=y[idx])
    clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=-1)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    metrics = {"overall_accuracy": round(accuracy_score(yte, pred), 3),
               "kappa": round(cohen_kappa_score(yte, pred), 3)}
    full = clf.predict(X).reshape(labels.shape)
    return full, metrics, clf


# ---- change detection ------------------------------------------------------
def change_matrix(lc_a, lc_b, class_names):
    """From-to transition matrix between two classified maps (pixel counts)."""
    import pandas as pd
    k = len(class_names)
    m = np.zeros((k, k), int)
    for i in range(k):
        for j in range(k):
            m[i, j] = int(((lc_a == i) & (lc_b == j)).sum())
    return pd.DataFrame(m, index=class_names, columns=class_names)


# ---- surface urban heat island --------------------------------------------
def suhi(lst_c, lc, urban_classes, rural_classes):
    """SUHI intensity = mean urban LST - mean rural LST (deg C)."""
    u = lst_c[np.isin(lc, urban_classes)].mean()
    r = lst_c[np.isin(lc, rural_classes)].mean()
    return {"urban_mean_C": round(float(u), 2),
            "rural_mean_C": round(float(r), 2),
            "SUHI_C": round(float(u - r), 2)}


def zonal_stats(values, zones):
    """Mean of `values` per integer zone id — the bridge to Chapter 3."""
    import pandas as pd
    return pd.DataFrame([{"zone": int(z),
                          "mean": round(float(values[zones == z].mean()), 3),
                          "n_px": int((zones == z).sum())}
                         for z in np.unique(zones)])


if __name__ == "__main__":
    a = sys.argv[1] if len(sys.argv) > 1 else "data/scene_2015.tif"
    b = sys.argv[2] if len(sys.argv) > 2 else "data/scene_2023.tif"
    s1, _ = read_scene(a)
    s2, _ = read_scene(b)
    print("NDVI  2015 mean %.3f -> 2023 %.3f" % (ndvi(s1).mean(), ndvi(s2).mean()))
    print("NDBI  2015 mean %.3f -> 2023 %.3f" % (ndbi(s1).mean(), ndbi(s2).mean()))
    print("LST   2015 mean %.2f C -> 2023 %.2f C" % (s1["lst_c"].mean(), s2["lst_c"].mean()))
