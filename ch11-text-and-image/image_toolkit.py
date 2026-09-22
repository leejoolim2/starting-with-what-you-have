"""
image_toolkit.py
----------------
Global South Urban Methods — Chapter 10 (image half).
Street-view analysis that runs on a laptop with no GPU: the Green View Index,
simple colour/texture features, and a classical classifier.

Production work uses pretrained CNNs or vision-language models. This module is
what you use when you cannot download a 400 MB model or wait on a GPU — and it
is enough for the two things urban researchers most often need: a green
exposure measure, and a coarse street-type classification.

    python image_toolkit.py

Requires: pillow, numpy, scikit-learn, pandas
"""
import numpy as np
import pandas as pd
from PIL import Image


# ============================================================ 1. GREEN VIEW INDEX
def green_view_index(path_or_arr, exg_threshold=12):
    """Green View Index — the share of the image that is vegetation.

    Uses Excess Green (ExG = 2G - R - B), a standard vegetation index for
    ordinary RGB photographs where no near-infrared band exists. This is the
    same logic as NDVI in Chapter 6, adapted to a consumer camera.
    """
    a = np.asarray(Image.open(path_or_arr) if isinstance(path_or_arr, str)
                   else path_or_arr, float)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    exg = 2 * g - r - b
    return float((exg > exg_threshold).mean())


def sky_view_factor(path_or_arr, brightness=140, blue_excess=12):
    """Rough sky fraction — a proxy for enclosure and daylight access."""
    a = np.asarray(Image.open(path_or_arr) if isinstance(path_or_arr, str)
                   else path_or_arr, float)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    return float(((b - r > blue_excess) & (b > brightness)).mean())


# ================================================================ 2. FEATURES
def image_features(path):
    """Colour histogram + edge density + the two indices above.

    Edge density separates regular facades from irregular, self-built fabric
    without any training data.
    """
    im = Image.open(path).convert("RGB")
    a = np.asarray(im, float)
    feats = []
    for c in range(3):                                   # 8-bin colour histogram
        h, _ = np.histogram(a[..., c], bins=8, range=(0, 255), density=True)
        feats.extend(h)
    grey = a.mean(-1)
    gx = np.abs(np.diff(grey, axis=1)).mean()
    gy = np.abs(np.diff(grey, axis=0)).mean()
    feats += [gx, gy, grey.std(),
              green_view_index(a), sky_view_factor(a)]
    return np.array(feats, float)


FEATURE_NAMES = ([f"{c}_hist_{i}" for c in "RGB" for i in range(8)] +
                 ["edge_h", "edge_v", "contrast", "GVI", "sky_fraction"])


def build_feature_table(label_csv):
    df = pd.read_csv(label_csv)
    X = np.vstack([image_features(f) for f in df.file])
    return X, df.label.values, df


# ============================================================ 3. CLASSIFICATION
def evaluate(X, y, k=5, seed=0):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    clf = RandomForestClassifier(n_estimators=400, random_state=seed, n_jobs=-1)
    cv = StratifiedKFold(k, shuffle=True, random_state=seed)
    s = cross_val_score(clf, X, y, cv=cv, scoring="f1_macro")
    return {"f1_macro_mean": round(float(s.mean()), 3),
            "f1_macro_std": round(float(s.std()), 3)}


def importance(X, y, names=FEATURE_NAMES, seed=0):
    from sklearn.ensemble import RandomForestClassifier
    clf = RandomForestClassifier(n_estimators=400, random_state=seed,
                                 n_jobs=-1).fit(X, y)
    return (pd.DataFrame({"feature": names, "importance": clf.feature_importances_})
            .sort_values("importance", ascending=False).reset_index(drop=True))


if __name__ == "__main__":
    X, y, df = build_feature_table("data/streetview_labels.csv")
    df["GVI"] = X[:, FEATURE_NAMES.index("GVI")]
    print(df.groupby("label")["GVI"].agg(["mean", "std"]).round(4).to_string())
    print("\nclassification:", evaluate(X, y))
    print("\ntop features:\n", importance(X, y).head(6).to_string(index=False))
