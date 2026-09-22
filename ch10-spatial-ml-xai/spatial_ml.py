"""
spatial_ml.py
-------------
Global South Urban Methods — Chapter 9 companion module.
Tree-based spatial models with honest validation and SHAP interpretation.

The single most important function here is `spatial_block_cv`. Random k-fold
cross-validation leaks information across nearby points and inflates the
reported skill of any spatial model. Block CV does not.

    python spatial_ml.py

Requires: scikit-learn, xgboost, shap, pandas, numpy, matplotlib
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.linear_model import LinearRegression


# ================================================== 1. VALIDATION THAT IS HONEST
def spatial_blocks(coords, n_blocks=5, seed=0):
    """Assign points to spatial blocks by tiling the study area into a grid,
    then randomly allocating whole tiles to folds."""
    xy = np.asarray(coords, float)
    side = int(np.ceil(np.sqrt(n_blocks * 4)))            # 4 tiles per fold
    xi = np.clip(((xy[:, 0] - xy[:, 0].min()) /
                  (np.ptp(xy[:, 0]) + 1e-9) * side).astype(int), 0, side-1)
    yi = np.clip(((xy[:, 1] - xy[:, 1].min()) /
                  (np.ptp(xy[:, 1]) + 1e-9) * side).astype(int), 0, side-1)
    tile = xi * side + yi
    rng = np.random.default_rng(seed)
    uniq = np.unique(tile)
    assign = dict(zip(uniq, rng.permutation(len(uniq)) % n_blocks))
    return np.array([assign[t] for t in tile])


def spatial_block_cv(model, X, y, coords, n_blocks=5, seed=0):
    """Leave-one-block-out CV. Reports the score you should publish."""
    folds = spatial_blocks(coords, n_blocks, seed)
    r2s, maes = [], []
    for f in range(n_blocks):
        tr, te = folds != f, folds == f
        if te.sum() < 5:
            continue
        m = _clone(model).fit(X[tr], y[tr])
        p = m.predict(X[te])
        r2s.append(r2_score(y[te], p))
        maes.append(mean_absolute_error(y[te], p))
    return {"r2_mean": round(float(np.mean(r2s)), 3),
            "r2_std": round(float(np.std(r2s)), 3),
            "mae_mean": round(float(np.mean(maes)), 3),
            "n_folds": len(r2s)}


def random_cv(model, X, y, k=5, seed=0):
    """Ordinary k-fold CV — shown only so you can see how much it overstates."""
    kf = KFold(k, shuffle=True, random_state=seed)
    r2s, maes = [], []
    for tr, te in kf.split(X):
        m = _clone(model).fit(X[tr], y[tr])
        p = m.predict(X[te])
        r2s.append(r2_score(y[te], p))
        maes.append(mean_absolute_error(y[te], p))
    return {"r2_mean": round(float(np.mean(r2s)), 3),
            "r2_std": round(float(np.std(r2s)), 3),
            "mae_mean": round(float(np.mean(maes)), 3)}


def _clone(model):
    from sklearn.base import clone
    return clone(model)


# ================================================================== 2. MODELS
def fit_models(X, y, seed=0):
    """Baseline OLS, random forest, and gradient boosting."""
    models = {
        "OLS": LinearRegression(),
        "RandomForest": RandomForestRegressor(
            n_estimators=400, min_samples_leaf=2, random_state=seed, n_jobs=-1),
    }
    try:
        from xgboost import XGBRegressor
        models["XGBoost"] = XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=5,
            subsample=0.85, colsample_bytree=0.85, random_state=seed,
            n_jobs=-1, verbosity=0)
    except ImportError:
        pass
    return {k: m.fit(X, y) for k, m in models.items()}


def add_coordinates(X, coords):
    """Adding x/y as features lets a tree model absorb spatial structure.
    Cheap and effective, but the coordinates then soak up the effect of any
    omitted spatially-patterned variable — interpret with that in mind."""
    return np.hstack([np.asarray(X, float), np.asarray(coords, float)])


# ============================================================ 3. INTERPRETATION
def shap_values(model, X, feature_names, max_display=None):
    """TreeSHAP contributions. Returns (values, explainer)."""
    import shap
    ex = shap.TreeExplainer(model)
    sv = ex.shap_values(X)
    imp = (pd.DataFrame({"feature": feature_names,
                         "mean_abs_shap": np.abs(sv).mean(0)})
           .sort_values("mean_abs_shap", ascending=False)
           .reset_index(drop=True))
    imp["mean_abs_shap"] = imp["mean_abs_shap"].round(4)
    return sv, imp


def local_shap_map(sv, feature_names, feature, coords):
    """Per-location SHAP for one feature — the ML analogue of a GWR
    local coefficient surface (Chapter 4)."""
    j = list(feature_names).index(feature)
    xy = np.asarray(coords, float)
    return pd.DataFrame({"x": xy[:, 0], "y": xy[:, 1],
                         "shap": sv[:, j]})


def partial_dependence(model, X, feature_idx, grid=40):
    """Average model response as one feature is swept across its range."""
    X = np.asarray(X, float)
    xs = np.linspace(np.percentile(X[:, feature_idx], 2),
                     np.percentile(X[:, feature_idx], 98), grid)
    out = []
    Xc = X.copy()
    for v in xs:
        Xc[:, feature_idx] = v
        out.append(model.predict(Xc).mean())
    return xs, np.array(out)


if __name__ == "__main__":
    df = pd.read_csv("data/urban_heat.csv")
    feats = ["ndvi", "bld_density", "dist_water_m", "impervious", "elevation_m"]
    X, y = df[feats].values, df["lst_c"].values
    coords = df[["x", "y"]].values

    models = fit_models(X, y)
    rf = models["RandomForest"]
    print("random k-fold CV  :", random_cv(rf, X, y))
    print("spatial block CV  :", spatial_block_cv(rf, X, y, coords))
    sv, imp = shap_values(rf, X, feats)
    print("\nSHAP importance:\n", imp.to_string(index=False))
