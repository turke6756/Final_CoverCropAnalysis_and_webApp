
"""
Reward Metrics + Trend + Clustering Pipeline (v3.9.1)
-----------------------------------------------------
- Accepts separate WINTER (SAVI/BSI) and CMI CSVs; merges on (season, orch_id).
- Computes rewards timeseries + robust seasonal trends (slopes/intercepts/R²).
- R²-based shrink on features (sqrt/linear/logistic) incl. optional level shrink.
- Trains on high-quality subset; **Phase 2 assigns ALL feasible orchards**.
- Optional CMI-only clustering (Young/Mid/Old) for maturity stratification.
- Exports cluster template (JSON; **JSON-safe**), coverage, flips, transitions.
- **Bakes in viz_table.csv** for the web app (no separate helper needed).

Default "cover2" features: [z_slope_Intensity, z_level_Intensity_t5]
"""

from __future__ import annotations
import os, re, json, warnings
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, silhouette_samples, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict

# -----------------------------
# JSON-safe helper
# -----------------------------

def _json_safe(obj):
    """Recursively convert numpy types and non-string dict keys for JSON serialization."""
    import numpy as _np
    if isinstance(obj, dict):
        new = {}
        for k, v in obj.items():
            if isinstance(k, (_np.generic,)):
                k = k.item()
            k = str(k)  # JSON keys must be strings
            new[k] = _json_safe(v)
        return new
    elif isinstance(obj, (list, tuple, set)):
        return [ _json_safe(x) for x in obj ]
    elif isinstance(obj, _np.ndarray):
        return obj.tolist()
    elif isinstance(obj, _np.generic):
        return obj.item()
    else:
        return obj

# -----------------------------
# Config / columns
# -----------------------------

COL_ORCH = "orch_id"
COL_SEASON = "season"
COL_PERIOD = "time_period"

COL_SAVI_MEAN = "orchard_savi_mean"
COL_SAVI_STD  = "orchard_savi_stddev"
COL_BSI_MEAN  = "orchard_bsi_mean"
COL_BSI_STD   = "orchard_bsi_stddev"

COL_VALID_FRAC = "valid_pixel_fraction"  # alias: valid_fraction

DEFAULT_TREND_PERIODS = [3,4,5,6,7]
EPS = 1e-6

# -----------------------------
# Small utils
# -----------------------------

def clip01(x): 
    return np.minimum(1.0, np.maximum(0.0, x))

def robust_z_to_01(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    q75 = np.nanpercentile(x, 75)
    q25 = np.nanpercentile(x, 25)
    iqr = max(q75 - q25, EPS)
    z = (x - med) / iqr
    z = np.clip(z, -3.0, 3.0)
    return (z + 3.0) / 6.0

def robust_z(x: np.ndarray) -> np.ndarray:
    med = np.nanmedian(x)
    q75 = np.nanpercentile(x, 75); q25 = np.nanpercentile(x, 25)
    iqr = max(q75 - q25, EPS)
    z = (x - med) / iqr
    return np.clip(z, -3.0, 3.0)

def high_tail01(u: np.ndarray) -> np.ndarray:
    return clip01((u - 0.5) / 0.3)

def period_weight(t: np.ndarray) -> np.ndarray:
    t = np.array(t, dtype=float)
    t = np.clip(t, 2, 7)
    return 0.3 + 0.4 * (t - 2.0) / (7.0 - 2.0)

def is_missing_row(row: pd.Series) -> bool:
    return (
        (abs(row.get(COL_SAVI_MEAN, 0.0)) <= EPS) and
        (abs(row.get(COL_BSI_MEAN , 0.0)) <= EPS) and
        (abs(row.get(COL_SAVI_STD , 0.0)) <= EPS) and
        (abs(row.get(COL_BSI_STD  , 0.0)) <= EPS)
    )

def weighted_linregress(x: np.ndarray, y: np.ndarray, w: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if w is None:
        w = np.ones_like(y, dtype=float)
    else:
        w = np.asarray(w, dtype=float)
        w = np.where(w > 0, w, EPS)

    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(w)
    x, y, w = x[mask], y[mask], w[mask]
    if x.size < 2:
        return (np.nan, np.nan, np.nan)

    m, b = np.polyfit(x, y, 1, w=w)
    yhat = m * x + b
    ybar = np.sum(w * y) / np.sum(w)
    sse = np.sum(w * (y - yhat) ** 2)
    sst = np.sum(w * (y - ybar) ** 2)
    r2 = 1.0 - (sse / (sst + EPS))
    return (float(m), float(b), float(r2))

def infer_season_from_path(path: str) -> Optional[int]:
    m = re.findall(r'(20\d{2})', os.path.basename(path))
    for s in m:
        y = int(s)
        if 2010 <= y <= 2035:
            return y
    return None

# -----------------------------
# Viz helpers (mirroring web app expectations)
# -----------------------------

ROLE_COLORS = {
    "Cover Crops":  "#2ca25f",  # cover (green)
    "Mixed":        "#6a51a3",  # optional 3rd class
    "Bare Soil":    "#8B4513",  # non-cover (brown)
}

def season_sort_from_key(season_str: str):
    if pd.isna(season_str): return np.nan
    s = str(season_str)
    if "_" in s:
        try:
            return int(s.split("_")[0])
        except ValueError:
            pass
    try:
        return int(s)
    except ValueError:
        return np.nan

def season_mid_from_key(season_str: str, mmdd: str = "-01-31"):
    if pd.isna(season_str): return ""
    s = str(season_str)
    if "_" in s:
        parts = s.split("_")
        if len(parts) == 2 and parts[1].isdigit():
            return f"{parts[1]}{mmdd}"
    digits = "".join([ch for ch in s if ch.isdigit()])
    yr = digits[:4] if len(digits) >= 4 else s
    return f"{yr}{mmdd}"

def map_label_to_role_and_color(label: str):
    lab = (label or "").strip()
    lower = lab.lower()
    if lower in {"cover", "likely cover cropping"}:
        role = "Cover Crops"
    elif lower in {"nocover", "no cover", "likely not cover cropping"}:
        role = "Bare Soil"
    elif lower in {"baseline", "mixed"}:
        role = "Mixed"
    else:
        role = "Bare Soil"
    return role, ROLE_COLORS.get(role, "#9ca3af")

def robust_minmax(v: np.ndarray, lo=10, hi=90):
    v = np.asarray(v, float)
    if v.size == 0 or not np.isfinite(v).any():
        return np.zeros_like(v, float)
    lo_v = np.nanpercentile(v, lo)
    hi_v = np.nanpercentile(v, hi)
    rng = hi_v - lo_v
    if not np.isfinite(rng) or rng <= 1e-9:
        return np.zeros_like(v, float)
    out = (v - lo_v) / rng
    return np.clip(out, 0, 1)

# -----------------------------
# CMI helpers
# -----------------------------

def summarize_cmi_table(df: pd.DataFrame) -> pd.DataFrame:
    keys = [COL_SEASON, COL_ORCH]
    meta_cols = [c for c in df.columns if c not in keys]
    num = df[meta_cols].select_dtypes(include=[np.number]).columns.tolist()
    if not num:
        return pd.DataFrame(columns=keys)
    agg = df.groupby(keys)[num].median().reset_index()
    rename = {c: f"CMI__{c}" for c in agg.columns if c not in keys}
    return agg.rename(columns=rename)

# -----------------------------
# Rewards & trends
# -----------------------------

def compute_rewards_timeseries(df: pd.DataFrame, strata: Optional[List[str]] = None) -> pd.DataFrame:
    work = df.copy()
    if (COL_VALID_FRAC not in work.columns) and ("valid_fraction" in work.columns):
        work[COL_VALID_FRAC] = work["valid_fraction"]

    for col in [COL_ORCH, COL_PERIOD, COL_SAVI_MEAN, COL_SAVI_STD, COL_BSI_MEAN, COL_BSI_STD]:
        if col not in work.columns:
            raise KeyError(f"Missing required column: {col}")

    if COL_SEASON not in work.columns:
        if "startyear" in work.columns:
            work[COL_SEASON] = work["startyear"]
        else:
            if "__source_csv" in work.columns:
                inferred = infer_season_from_path(str(work["__source_csv"].iloc[0]))
                work[COL_SEASON] = inferred if inferred is not None else 0
            else:
                work[COL_SEASON] = 0

    work["row_is_missing"] = work.apply(is_missing_row, axis=1)
    work = work.loc[~work["row_is_missing"]].copy()

    work["SAVI_rz"] = np.nan
    work["BSI_green"] = np.nan
    work["Intensity"] = np.nan
    work["Fill"] = np.nan
    work["Uniformity"] = np.nan

    work["CV_SAVI"] = work[COL_SAVI_STD] / (np.abs(work[COL_SAVI_MEAN]) + EPS)
    work["CV_BSI"]  = work[COL_BSI_STD ] / (np.abs(work[COL_BSI_MEAN ]) + EPS)

    keys = [COL_SEASON]
    if strata:
        for s in strata:
            if s not in work.columns: 
                raise KeyError(f"Stratum column {s} not found.")
        keys.extend(strata)

    def per_group(g: pd.DataFrame) -> pd.DataFrame:
        def scale_periodwise(col: str, invert: bool=False):
            vals = []
            for tp, sub in g.groupby(COL_PERIOD):
                rz = robust_z_to_01(sub[col].values)
                if invert: rz = 1.0 - rz
                vals.append(pd.Series(rz, index=sub.index))
            return pd.concat(vals).sort_index()

        savi_rz = scale_periodwise(COL_SAVI_MEAN, invert=False)
        bsi_green = scale_periodwise(COL_BSI_MEAN, invert=True)

        g = g.copy()
        g["SAVI_rz"] = savi_rz.values
        g["BSI_green"] = bsi_green.values

        w = period_weight(g[COL_PERIOD].values)
        g["Intensity"] = w * g["SAVI_rz"].values + (1.0 - w) * g["BSI_green"].values
        g["Fill"] = 0.5 * high_tail01(g["SAVI_rz"].values) + 0.5 * high_tail01(g["BSI_green"].values)

        def scale_cv(col: str) -> pd.Series:
            vals = []
            for tp, sub in g.groupby(COL_PERIOD):
                rz = robust_z_to_01(sub[col].values)
                inv = 1.0 - rz
                vals.append(pd.Series(inv, index=sub.index))
            return pd.concat(vals).sort_index()

        cv_savi_up = scale_cv("CV_SAVI")
        cv_bsi_up  = scale_cv("CV_BSI")
        g["Uniformity"] = 0.5 * cv_savi_up.values + 0.5 * cv_bsi_up.values
        return g

    out = work.groupby(keys, group_keys=False).apply(per_group).reset_index(drop=True)
    return out

def compute_trends(rewards_df: pd.DataFrame, trend_periods: Optional[List[int]] = None,
                   use_valid_fraction: bool = True) -> pd.DataFrame:
    if trend_periods is None: 
        trend_periods = DEFAULT_TREND_PERIODS

    def weight_row(row: pd.Series) -> float:
        if use_valid_fraction and (COL_VALID_FRAC in rewards_df.columns):
            v = row.get(COL_VALID_FRAC, np.nan)
            if pd.notna(v) and v > 0: 
                return float(v)
        return 1.0 / (row.get(COL_SAVI_STD, 0.0) + row.get(COL_BSI_STD, 0.0) + EPS)

    rewards_long = rewards_df.copy()
    rewards = ["Intensity", "Fill", "Uniformity"]
    recs = []

    for (season, orch), sub in rewards_long.groupby([COL_SEASON, COL_ORCH]):
        sub = sub[sub[COL_PERIOD].isin(trend_periods)].copy()
        if sub.empty:
            recs.append({COL_SEASON: season, COL_ORCH: orch, "n_periods_used": 0,
                         **{f"slope_{r}": np.nan for r in rewards},
                         **{f"intercept_{r}": np.nan for r in rewards},
                         **{f"r2_{r}": np.nan for r in rewards}})
            continue
        w = sub.apply(weight_row, axis=1).values
        t = sub[COL_PERIOD].values.astype(float)

        row = {COL_SEASON: season, COL_ORCH: orch, "n_periods_used": int(sub.shape[0])}
        for r in rewards:
            y = sub[r].values.astype(float)
            m, b, r2 = weighted_linregress(t, y, w=w)
            row[f"slope_{r}"] = m
            row[f"intercept_{r}"] = b
            row[f"r2_{r}"] = r2
        recs.append(row)

    trends = pd.DataFrame(recs)

    # Robust seasonal z-normalization of slopes
    for r in ["Intensity","Fill","Uniformity"]:
        col = f"slope_{r}"
        zvals = []
        for s, g in trends.groupby(COL_SEASON):
            x = g[col].values
            z = robust_z(x)
            zvals.append(pd.Series(z, index=g.index))
        trends[f"z_{col}"] = pd.concat(zvals).sort_index()

    # Predicted Intensity at t=5 for level feature
    trends["pred_Intensity_t5"] = trends["intercept_Intensity"] + 5.0 * trends["slope_Intensity"]
    zlev = []
    for s, g in trends.groupby(COL_SEASON):
        x = g["pred_Intensity_t5"].values
        z = robust_z(x)
        zlev.append(pd.Series(z, index=g.index))
    trends["z_level_Intensity_t5"] = pd.concat(zlev).sort_index()

    return trends

# -----------------------------
# IO helpers
# -----------------------------

def load_winter_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["__source_csv"] = path
    if (COL_VALID_FRAC not in df.columns) and ("valid_fraction" in df.columns):
        df[COL_VALID_FRAC] = df["valid_fraction"]
    if COL_SEASON not in df.columns:
        if "startyear" in df.columns:
            df[COL_SEASON] = df["startyear"]
        else:
            guess = infer_season_from_path(path) or 0
            df[COL_SEASON] = guess
    return df

def load_cmi_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["__source_csv"] = path
    if COL_SEASON not in df.columns:
        if "startyear" in df.columns:
            df[COL_SEASON] = df["startyear"]
        else:
            guess = infer_season_from_path(path) or 0
            df[COL_SEASON] = guess
    return df

def run_pipeline_inputs(csv_paths: List[str], winter_paths: List[str], cmi_paths: List[str],
                        strata: Optional[List[str]] = None,
                        trend_periods: Optional[List[int]] = None,
                        use_valid_fraction: bool = True) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if csv_paths:
        dfs = [load_winter_csv(p) for p in csv_paths]
    else:
        dfs = [load_winter_csv(p) for p in winter_paths]
    raw_winter = pd.concat(dfs, ignore_index=True)

    rewards_ts = compute_rewards_timeseries(raw_winter, strata=strata)
    trends = compute_trends(rewards_ts, trend_periods=trend_periods, use_valid_fraction=use_valid_fraction)

    if cmi_paths:
        cdfs = [load_cmi_csv(p) for p in cmi_paths]
        raw_cmi = pd.concat(cdfs, ignore_index=True)
        cmi_lookup = summarize_cmi_table(raw_cmi)
    else:
        cmi_lookup = pd.DataFrame(columns=[COL_SEASON, COL_ORCH])

    rewards_ts = rewards_ts.merge(cmi_lookup, on=[COL_SEASON, COL_ORCH], how="left")
    trends = trends.merge(cmi_lookup, on=[COL_SEASON, COL_ORCH], how="left")
    return raw_winter, rewards_ts, trends, cmi_lookup

# -----------------------------
# Features + shrink
# -----------------------------

def build_features(trends: pd.DataFrame, feature_set: str = "cover2") -> Tuple[pd.DataFrame, List[str]]:
    t = trends.copy()
    if feature_set == "slopes":
        cols = ["z_slope_Intensity", "z_slope_Fill", "z_slope_Uniformity"]
        return t, cols
    elif feature_set == "greenup3":
        t["green_slope"] = 0.5 * (t["z_slope_Intensity"] + t["z_slope_Fill"])
        t["green_level"] = t["z_level_Intensity_t5"]
        t["uniformity_slope"] = t["z_slope_Uniformity"]
        cols = ["green_slope", "green_level", "uniformity_slope"]
        return t, cols
    elif feature_set == "cover2":
        cols = ["z_slope_Intensity", "z_level_Intensity_t5"]
        return t, cols
    else:
        raise ValueError(f"Unknown feature_set: {feature_set}")

def shrink_factor(r2: np.ndarray, mode: str) -> np.ndarray:
    r2 = np.clip(r2, 0.0, 1.0)
    if mode == "none":
        return np.ones_like(r2)
    if mode == "sqrt":
        return np.sqrt(r2)
    if mode == "linear":
        return clip01((r2 - 0.2) / 0.6)
    if mode == "logistic":
        return 1.0 / (1.0 + np.exp(-(r2 - 0.5) / 0.1))
    raise ValueError(f"Unknown shrink mode: {mode}")

def apply_shrink(feats_df: pd.DataFrame, feat_cols: List[str], trends: pd.DataFrame,
                 shrink_mode: str, shrink_level: bool, feature_one: Optional[str]) -> Tuple[pd.DataFrame, List[str]]:
    if shrink_mode == "none":
        feats_df["shrink_mode"] = "none"
        return feats_df, feat_cols

    f = feats_df.copy()
    if feature_one is not None:
        mapping_col = {"intensity_slope": "z_slope_Intensity","fill_slope": "z_slope_Fill","uniformity_slope": "z_slope_Uniformity"}
        mapping_r2  = {"intensity_slope": "r2_Intensity","fill_slope": "r2_Fill","uniformity_slope": "r2_Uniformity"}
        col, r2col = mapping_col[feature_one], mapping_r2[feature_one]
        sf = shrink_factor(trends.loc[f.index, r2col].values, shrink_mode)
        f[col] = f[col].values * sf
        f["shrink_mode"] = shrink_mode
        return f, feat_cols

    if "green_slope" in feat_cols:
        r2_green = np.nanmean(np.vstack([trends["r2_Intensity"].values, trends["r2_Fill"].values]), axis=0)
        sf = shrink_factor(r2_green, shrink_mode)
        f["green_slope"] = f["green_slope"].values * sf
        if shrink_level:
            sfL = shrink_factor(trends["r2_Intensity"].values, shrink_mode)
            f["green_level"] = f["green_level"].values * sfL
        f["shrink_mode"] = shrink_mode
        return f, feat_cols

    if "z_slope_Intensity" in feat_cols:
        sfI = shrink_factor(trends["r2_Intensity"].values, shrink_mode)
        f["z_slope_Intensity"] = f["z_slope_Intensity"].values * sfI
    if shrink_level and "z_level_Intensity_t5" in feat_cols:
        sfL = shrink_factor(trends["r2_Intensity"].values, shrink_mode)
        f["z_level_Intensity_t5"] = f["z_level_Intensity_t5"].values * sfL
    if "z_slope_Fill" in feat_cols:
        sfF = shrink_factor(trends["r2_Fill"].values, shrink_mode)
        f["z_slope_Fill"] = f["z_slope_Fill"].values * sfF
    if "z_slope_Uniformity" in feat_cols:
        sfU = shrink_factor(trends["r2_Uniformity"].values, shrink_mode)
        f["z_slope_Uniformity"] = f["z_slope_Uniformity"].values * sfU

    f["shrink_mode"] = shrink_mode
    return f, feat_cols

# -----------------------------
# K choice & model fit
# -----------------------------

def choose_k_by_silhouette(X: np.ndarray, method: str, ks: List[int], random_state: int = 42) -> Tuple[int, Dict[int, float]]:
    sils = {}
    for k in ks:
        if X.shape[0] <= k: 
            continue
        if method == "kmeans":
            km = KMeans(n_clusters=k, n_init=25, random_state=random_state)
            labels = km.fit_predict(X)
        else:
            gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=random_state, n_init=5, reg_covar=1e-6)
            labels = gmm.fit_predict(X)
        sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else np.nan
        sils[k] = sil
    if not sils:
        return None, {}
    best_k = max(sils, key=lambda k: (np.nan_to_num(sils[k], nan=-1.0), -k))
    return best_k, sils

def choose_k_by_bic_gmm(X: np.ndarray, ks: List[int], random_state: int = 42) -> Tuple[int, Dict[int, float], Dict[int, float]]:
    bics, aics = {}, {}
    for k in ks:
        if X.shape[0] <= k: 
            continue
        gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=random_state, n_init=5, reg_covar=1e-6)
        gmm.fit(X)
        bics[k], aics[k] = gmm.bic(X), gmm.aic(X)
    if not bics:
        return None, {}, {}
    best_k = min(bics, key=lambda k: bics[k])
    return best_k, bics, aics

def fit_cluster_model(X: np.ndarray, method: str, k: int, random_state: int = 42):
    if method == "kmeans":
        model = KMeans(n_clusters=k, n_init=50, random_state=random_state)
        labels = model.fit_predict(X)
        proba = None
        centers = model.cluster_centers_
        return model, labels, proba, centers
    else:
        gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=random_state, n_init=5, reg_covar=1e-6)
        model = gmm.fit(X)
        labels = model.predict(X)
        proba = model.predict_proba(X)
        centers = model.means_
        return model, labels, proba, centers

# -----------------------------
# Labeling & summaries
# -----------------------------

def label_clusters_cover2(centers: np.ndarray, schema: str = "short") -> Dict[int,str]:
    scores = centers[:,0] + 0.8*centers[:,1]
    order = np.argsort(scores)  # low -> high
    mapping = {}
    if len(centers) == 2:
        if schema == "likely":
            mapping[int(order[0])] = "Likely Not Cover Cropping"
            mapping[int(order[1])] = "Likely Cover Cropping"
        else:
            mapping[int(order[0])] = "NoCover"
            mapping[int(order[1])] = "Cover"
    elif len(centers) == 3:
        if schema == "likely":
            mapping[int(order[0])] = "Likely Not Cover Cropping"
            mapping[int(order[1])] = "Likely Baseline"
            mapping[int(order[2])] = "Likely Cover Cropping"
        else:
            mapping[int(order[0])] = "NoCover"
            mapping[int(order[1])] = "Baseline"
            mapping[int(order[2])] = "Cover"
    else:
        for rank, c in enumerate(order):
            mapping[int(c)] = f"Cluster{rank}"
    return mapping

def compute_cluster_profiles(assign_df: pd.DataFrame, features: List[str], trends: pd.DataFrame,
                             labels: np.ndarray, X: np.ndarray) -> pd.DataFrame:
    sil_samp = silhouette_samples(X, labels) if len(np.unique(labels)) > 1 else np.full(len(labels), np.nan)
    assign_df = assign_df.copy()
    assign_df["silhouette"] = sil_samp
    profs = []
    for c, g in assign_df.groupby("cluster"):
        row = {"cluster": int(c), "n": int(len(g))}
        for f in features:
            row[f"centroid_{f}"] = float(np.nanmean(g[f].values))
        row["mean_silhouette"] = float(np.nanmean(g["silhouette"].values))
        row["uncertain_rate"] = float(np.mean(g.get("uncertain", pd.Series(np.zeros(len(g)))).values))
        merged = g[[COL_SEASON, COL_ORCH]].merge(trends, on=[COL_SEASON, COL_ORCH], how="left")
        for base in ["slope_Intensity","slope_Fill","slope_Uniformity",
                     "z_slope_Intensity","z_slope_Fill","z_slope_Uniformity",
                     "r2_Intensity","r2_Fill","r2_Uniformity","z_level_Intensity_t5"]:
            if base in merged.columns:
                vals = merged[base].values
                row[f"mean_{base}"] = float(np.nanmean(vals))
                row[f"p25_{base}"]  = float(np.nanpercentile(vals, 25)) if np.isfinite(vals).any() else np.nan
                row[f"p75_{base}"]  = float(np.nanpercentile(vals, 75)) if np.isfinite(vals).any() else np.nan
        for cmi in [c for c in merged.columns if c.startswith("CMI__")]:
            vals = merged[cmi].values
            row[f"mean_{cmi}"] = float(np.nanmean(vals))
        profs.append(row)
    return pd.DataFrame(profs)

# -----------------------------
# Phase 2 assignment helpers
# -----------------------------

def dim_normalized_dist(x: np.ndarray, center: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(center)
    if not mask.any():
        return np.inf
    d = np.linalg.norm((x[mask] - center[mask]))
    D = len(center); m = mask.sum()
    scale = np.sqrt(D / m)
    return float(d * scale)

def assign_by_centroids(X_all: np.ndarray, centers: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n, D = X_all.shape
    K = centers.shape[0]
    Dmat = np.full((n, K), np.nan, dtype=float)
    for i in range(n):
        xi = X_all[i, :]
        for k in range(K):
            Dmat[i, k] = dim_normalized_dist(xi, centers[k, :])
    cluster = np.nanargmin(Dmat, axis=1)
    sorted_d = np.sort(Dmat, axis=1)
    Dmin = sorted_d[:, 0]
    D2 = np.where(np.isfinite(sorted_d[:, 1]), sorted_d[:, 1], np.inf)
    return cluster, Dmin, D2

# -----------------------------
# CMI clustering
# -----------------------------

def cmi_cluster(trends: pd.DataFrame, cmi_col: str, method: str, kselect: str, ks: List[int]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    t = trends.copy()
    if cmi_col not in t.columns:
        pref = f"CMI__{cmi_col}"
        if pref in t.columns:
            cmi_col = pref
        else:
            raise KeyError(f"CMI column {cmi_col} not found in trends.")
    z_list = []
    for s, g in t.groupby(COL_SEASON):
        z = robust_z(g[cmi_col].values)
        z_list.append(pd.Series(z, index=g.index))
    t["CMI_z"] = pd.concat(z_list).sort_index()
    mask = np.isfinite(t["CMI_z"].values)
    X = t.loc[mask, ["CMI_z"]].values

    metrics_rows = []
    if kselect == "silhouette":
        sils = {}
        for k in ks:
            if X.shape[0] <= k: 
                continue
            km = KMeans(n_clusters=k, n_init=25, random_state=42)
            labels = km.fit_predict(X)
            sil = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else np.nan
            sils[k] = sil
        chosen_k = max(sils, key=lambda k: (np.nan_to_num(sils[k], nan=-1.0), -k)) if sils else None
        for k in ks: metrics_rows.append({"K": k, "silhouette": sils.get(k, np.nan)})
    else:
        bics = {}
        for k in ks:
            if X.shape[0] <= k: continue
            gmm = GaussianMixture(n_components=k, covariance_type="diag", random_state=42, n_init=5, reg_covar=1e-6)
            gmm.fit(X); bics[k] = gmm.bic(X)
        chosen_k = min(bics, key=lambda k: bics[k]) if bics else None
        for k, v in bics.items(): metrics_rows.append({"K": k, "bic": v})

    if chosen_k is None:
        return (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(metrics_rows))

    km = KMeans(n_clusters=chosen_k, n_init=50, random_state=42)
    labels = km.fit_predict(X)
    centers = km.cluster_centers_.reshape(-1)

    order = np.argsort(centers)
    label_map = {}
    if chosen_k == 2:
        label_map[int(order[0])] = "Young"; label_map[int(order[1])] = "Old"
    elif chosen_k == 3:
        label_map[int(order[0])] = "Young"; label_map[int(order[1])] = "Middle"; label_map[int(order[2])] = "Old"
    else:
        for i, c in enumerate(order): label_map[int(c)] = f"CMI_{i}"

    assign = t.loc[mask, [COL_SEASON, COL_ORCH, "CMI_z"]].copy()
    assign["cmi_cluster"] = labels
    assign["cmi_label"] = [label_map[int(i)] for i in labels]

    cents = pd.DataFrame({"cluster": np.arange(chosen_k), "CMI_z_center": centers})
    cents["cmi_label"] = cents["cluster"].map(lambda k: label_map[int(k)])

    metrics = pd.DataFrame(metrics_rows)
    metrics["chosen_k"] = chosen_k
    metrics["method"] = "kmeans"
    metrics["chosen_silhouette"] = silhouette_score(X, labels) if len(np.unique(labels)) > 1 else np.nan
    metrics["feature"] = "CMI_z"
    return assign, cents, metrics

# -----------------------------
# Viz table builder (integrated)
# -----------------------------

def ensure_softness(assign_df: pd.DataFrame) -> pd.Series:
    if "best_soft" in assign_df.columns and np.isfinite(assign_df["best_soft"]).any():
        s = assign_df["best_soft"].astype(float).clip(0, 1)
        return s.fillna(s.median() if np.isfinite(s.median()) else 0.5)
    if "silhouette" in assign_df.columns and np.isfinite(assign_df["silhouette"]).any():
        sil = assign_df["silhouette"].astype(float)
        s = (sil + 1.0) * 0.5
        return s.fillna(np.nanmedian(s) if np.isfinite(np.nanmedian(s)) else 0.5)
    zs =  assign_df["z_slope_Intensity"].values if "z_slope_Intensity" in assign_df.columns else np.zeros(len(assign_df))
    zl =  assign_df["z_level_Intensity_t5"].values if "z_level_Intensity_t5" in assign_df.columns else np.zeros(len(assign_df))
    score = zs + 0.8*zl
    s = robust_minmax(score, lo=5, hi=95)
    return pd.Series(s, index=assign_df.index)

def build_viz_table(assign_df: pd.DataFrame, trends_df: pd.DataFrame,
                    flips_df: Optional[pd.DataFrame] = None,
                    cmi_assign_df: Optional[pd.DataFrame] = None,
                    season_mid_mmdd: str = "-01-31") -> pd.DataFrame:
    df = assign_df.copy()

    for need in [COL_SEASON, COL_ORCH]:
        if need not in df.columns:
            raise KeyError(f"assign_df missing required column '{need}'")

    soft = ensure_softness(df); df["best_soft"] = soft.values

    keep_trend_cols = [COL_SEASON,COL_ORCH,"n_periods_used","z_level_Intensity_t5","CMI",
                       "slope_Intensity","r2_Intensity","z_slope_Intensity"]
    tr = trends_df[[c for c in keep_trend_cols if c in trends_df.columns]].copy()
    tr = tr.rename(columns={"n_periods_used":"npts_total","z_level_Intensity_t5":"env_z_season"})
    df = df.merge(tr, on=[COL_SEASON, COL_ORCH], how="left")

    if cmi_assign_df is not None and not cmi_assign_df.empty:
        m = cmi_assign_df[[COL_SEASON, COL_ORCH, "cmi_label"]].copy()
        m["stratum"] = m["cmi_label"].astype(str).str.title()
        # Keep the proper age classifications from CMI clustering
        m["stratum"] = m["stratum"].replace({"Middle":"Mid"})
        df = df.merge(m[[COL_SEASON, COL_ORCH, "stratum"]], on=[COL_SEASON, COL_ORCH], how="left")
    if "stratum" not in df.columns:
        df["stratum"] = "Mid"

    if "label" in df.columns:
        role_color = df["label"].apply(map_label_to_role_and_color)
    else:
        # Create a Series with empty strings for each row when label column is missing
        labels = pd.Series([""] * len(df), index=df.index)
        role_color = labels.apply(map_label_to_role_and_color)
    df["cluster_role"] = role_color.apply(lambda x: x[0])
    df["cluster_role_color"] = role_color.apply(lambda x: x[1])

    df["best_soft_norm"] = 0.0
    for (s, st), gidx in df.groupby([COL_SEASON,"stratum"]).groups.items():
        vals = df.loc[gidx, "best_soft"].values
        df.loc[gidx, "best_soft_norm"] = robust_minmax(vals, lo=10, hi=90)

    df["alpha_0_1"] = df["best_soft_norm"].clip(0,1)
    df["alpha_0_100"] = (df["alpha_0_1"] * 100).round().astype(int)

    df["season_sort"] = df[COL_SEASON].apply(season_sort_from_key).astype("Int64")
    df["season_mid"] = df[COL_SEASON].apply(lambda s: season_mid_from_key(s, season_mid_mmdd))

    df["streak_len_current"] = np.nan
    df["time_since_last_flip"] = np.nan
    df["flip_style_last"] = ""
    df["delta_last"] = np.nan

    if flips_df is not None and not flips_df.empty:
        fl = flips_df.copy()
        for need in ["orch_id","season_from","season_to","flip","magnitude","distance"]:
            if need not in fl.columns:
                if need.upper() in fl.columns: fl = fl.rename(columns={need.upper():need})
        by_orch = {k: g.sort_values("season_to") for k,g in fl.groupby(COL_ORCH)}
        for orch, g in df.groupby(COL_ORCH):
            g = g.sort_values("season_sort")
            seasons = g[COL_SEASON].tolist()
            sorts   = g["season_sort"].tolist()
            f = by_orch.get(orch, pd.DataFrame(columns=["season_to","flip","magnitude","distance"]))
            # Fix: Create index with season strings, not integers
            f_index = {str(r["season_to"]): r for _, r in f.iterrows() if pd.notna(r["season_to"])}
            last_flip_sort_idx = None
            streak_len = 0
            prev_role = None
            for i, (sea, srt) in enumerate(zip(seasons, sorts)):
                role = df.loc[g.index[i], "cluster_role"]
                if (prev_role is not None) and (role == prev_role):
                    streak_len += 1
                else:
                    streak_len = 1
                prev_role = role
                df.loc[g.index[i], "streak_len_current"] = streak_len

                # Fix: Look up using season string directly
                row_flip = f_index.get(str(sea), None)
                if row_flip is not None and int(row_flip.get("flip",0)) == 1:
                    last_flip_sort_idx = srt
                    df.loc[g.index[i], "time_since_last_flip"] = 0
                    df.loc[g.index[i], "flip_style_last"] = str(row_flip.get("magnitude",""))
                    df.loc[g.index[i], "delta_last"] = float(row_flip.get("distance", np.nan))
                else:
                    if last_flip_sort_idx is None:
                        df.loc[g.index[i], "time_since_last_flip"] = np.nan
                    else:
                        df.loc[g.index[i], "time_since_last_flip"] = int(srt - last_flip_sort_idx)

    df["confidence"] = df["best_soft"].round(3)
    df["p_cover"] = np.where(df["cluster_role"] == "Cover Crops", df["best_soft"], 1.0 - df["best_soft"])

    # Add consecutive cover crop streak calculations (1-8 years)
    for streak_years in range(1, 9):
        df[f"cover_streak_{streak_years}y"] = 0
    
    for orch, g in df.groupby(COL_ORCH):
        g = g.sort_values("season_sort")
        cover_roles = g["cluster_role"].values
        
        # Calculate consecutive cover crop streaks ending at each season
        for i, role in enumerate(cover_roles):
            if role == "Cover Crops":
                # Count backwards to find consecutive cover crop years
                consecutive_count = 1
                for j in range(i-1, -1, -1):
                    if cover_roles[j] == "Cover Crops":
                        consecutive_count += 1
                    else:
                        break
                
                # Mark streak achievements (1-8 years)
                for streak_years in range(1, min(9, consecutive_count + 1)):
                    df.loc[g.index[i], f"cover_streak_{streak_years}y"] = 1
    
    # Add county column placeholder for GeoJSON join
    df["county"] = ""

    out_cols = [
        "orch_id","season","season_mid","season_sort","stratum","cluster_role","cluster_role_color",
        "best_soft","best_soft_norm","alpha_0_1","alpha_0_100","streak_len_current","time_since_last_flip",
        "flip_style_last","delta_last","confidence","npts_total","env_z_season",
        "slope_Intensity","r2_Intensity","z_slope_Intensity",
        "label","silhouette","p_cover","county"
    ] + [f"cover_streak_{i}y" for i in range(1, 9)]
    for c in out_cols:
        if c not in df.columns:
            df[c] = np.nan

    out = df[out_cols].copy().sort_values(["season_sort","orch_id"])
    return out

# -----------------------------
# Main
# -----------------------------

def main():
    import argparse
    p = argparse.ArgumentParser(description="Rewards + Trends + Clustering (multi-CSV) with Phase-2 assignment, CMI, flips, and viz table")

    # Inputs
    p.add_argument("--csv", dest="csvs", nargs="+", default=None, help="Legacy: one or more WINTER CSVs (SAVI/BSI).")
    p.add_argument("--csv-winter", dest="csv_winter", nargs="+", default=None, help="WINTER CSVs (SAVI/BSI).")
    p.add_argument("--csv-cmi", dest="csv_cmi", nargs="+", default=None, help="CMI CSVs (orchard-season).")
    p.add_argument("--cmi-col", default=None, help="Name of the canonical CMI column in CMI CSV (auto-detect if omitted).")

    p.add_argument("--strata", nargs="*", default=None, help="Optional strata columns (e.g., CNTY).")
    p.add_argument("--periods", nargs="*", type=int, default=None, help="Trend periods (default 3 4 5 6 7).")
    p.add_argument("--no-valid-frac", action="store_true", help="Ignore valid_pixel_fraction even if present.")

    # Clustering options
    p.add_argument("--cluster-method", choices=["kmeans","gmm"], default="kmeans")
    p.add_argument("--k-select", choices=["silhouette","bic"], default="silhouette")
    p.add_argument("--ks", nargs="*", type=int, default=[2,3], help="Candidate K values (default 2 3).")
    p.add_argument("--feature-set", choices=["slopes","greenup3","cover2"], default="cover2")
    p.add_argument("--feature-one", choices=["intensity_slope","fill_slope","uniformity_slope"], default=None,
                   help="Override to 1D clustering on a single z-slope feature.")
    p.add_argument("--min-q", type=float, default=0.2, help="Quality filter threshold for training (default 0.2).")
    p.add_argument("--per-season", action="store_true", help="Compute per-season silhouettes with chosen K (refit per season for QA).")
    p.add_argument("--cover-label-schema", choices=["short","likely"], default="short", help="Naming for cover clusters.")

    # R² shrink options
    p.add_argument("--shrink-by-r2", choices=["none","sqrt","linear","logistic"], default="sqrt",
                   help="Shrink slopes by a function of R² to reduce noise influence (default sqrt).")
    p.add_argument("--shrink-level", action="store_true", help="Also shrink intensity level.")

    # Phase-2 ALL assignment thresholds
    p.add_argument("--assign-min-q", type=float, default=0.0, help="Min Q for Phase-2 assignment (default 0.0).")
    p.add_argument("--assign-min-periods", type=int, default=2, help="Min periods for trend fit to assign (default 2).")
    p.add_argument("--confidence-threshold", type=float, default=0.6, help="Uncertain if confidence below this (default 0.6).")
    p.add_argument("--distance-threshold", type=float, default=1.5, help="Uncertain if Dmin above this (z-units; default 1.5).")

    # CMI clustering (optional)
    p.add_argument("--cmi-cluster", action="store_true", help="Run 1D clustering on CMI (season-z).")
    p.add_argument("--cmi-method", choices=["kmeans"], default="kmeans")
    p.add_argument("--cmi-kselect", choices=["silhouette","bic"], default="silhouette")
    p.add_argument("--cmi-ks", nargs="*", type=int, default=[2,3], help="Candidate K for CMI clustering.")

    # Outputs
    p.add_argument("--out-ts", default="rewards_timeseries.csv", help="Output concatenated timeseries CSV.")
    p.add_argument("--out-trends", default="reward_trends.csv", help="Output concatenated trends CSV.")
    p.add_argument("--out-assign", default="cluster_assignments_train.csv", help="Training (Phase-1) cluster assignments.")
    p.add_argument("--out-assign-all", default="cluster_assignments_all.csv", help="ALL (Phase-2) cluster assignments.")
    p.add_argument("--out-centroids", default="cluster_centroids.csv", help="Cluster centers.")
    p.add_argument("--out-metrics", default="cluster_model_metrics.csv", help="Per-K metrics (silhouette/BIC).")
    p.add_argument("--out-season-sil", default="season_silhouette.csv", help="Silhouette per season (train subset).")
    p.add_argument("--out-profiles", default="cluster_profiles.csv", help="Per-cluster profile summary (train subset).")
    p.add_argument("--out-coverage", default="clustering_coverage.csv", help="Coverage tracking.")
    p.add_argument("--out-template", default="cluster_template.json", help="Frozen cluster template.")

    # CMI outputs
    p.add_argument("--out-cmi-analysis", default="cluster_cmi_analysis.csv", help="CMI vs cover cluster relationships.")
    p.add_argument("--out-cmi-assign", default="cmi_cluster_assignments.csv", help="CMI-only cluster assignments.")
    p.add_argument("--out-cmi-centroids", default="cmi_cluster_centroids.csv", help="CMI-only cluster centers.")
    p.add_argument("--out-cmi-metrics", default="cmi_cluster_model_metrics.csv", help="CMI-only model selection metrics.")
    p.add_argument("--out-crosstab", default="cluster_cmi_crosstab.csv", help="Cover × CMI label crosstab.")
    p.add_argument("--out-orchard-profiles", default="orchard_profiles.csv", help="Orchard-level majority labels over seasons.")

    # Flips/Transitions outputs
    p.add_argument("--out-flips", default="orchard_flips.csv", help="Orchard flip log between seasons (ALL assignments).")
    p.add_argument("--out-transitions", default="cluster_transitions.csv", help="Aggregated transitions (ALL assignments).")

    # Viz table
    p.add_argument("--out-viz", default="viz_table.csv", help="Single CSV for the web app.")
    p.add_argument("--season-mid-mmdd", default="-01-31", help="Midpoint date suffix for each winter season (default Jan 31).")

    
    p.add_argument("--out-attrs-dir", default="attrs", help="Directory for per-season attribute CSVs.")
    p.add_argument("--out-streaks",   default="streaks.csv", help="Time-agnostic streaks CSV.")
    p.add_argument("--out-manifest",  default="manifest.json", help="Manifest JSON for UI.")
    p.add_argument("--ui-uncertain-threshold", type=float, default=0.15,
                      help="UI threshold on best_soft_norm for 'uncertain' (default 0.15).")
    args = p.parse_args()

    # -----------------------------
    # Load + compute base tables
    # -----------------------------
    if not args.csvs and not args.csv_winter:
        raise ValueError("Provide --csv (legacy) or --csv-winter.")
    winter_paths = args.csvs if args.csvs else args.csv_winter
    cmi_paths = args.csv_cmi if args.csv_cmi else []

    # Run inputs
    raw_winter, rewards_ts, trends, cmi_lookup = run_pipeline_inputs(
        csv_paths=args.csvs,
        winter_paths=winter_paths,
        cmi_paths=cmi_paths,
        strata=args.strata,
        trend_periods=args.periods,
        use_valid_fraction=not args.no_valid_frac,
    )

    # Determine canonical CMI column (optional)
    cmi_col = args.cmi_col
    if cmi_col is None and not cmi_lookup.empty:
        candidates = [c for c in cmi_lookup.columns if c.startswith("CMI__")]
        pref = [c for c in candidates if "season_z" in c.lower()]
        cmi_col = pref[0] if pref else (candidates[0] if candidates else None)
    canonical_cmi = None
    if cmi_col:
        if cmi_col in rewards_ts.columns:
            canonical_cmi = cmi_col
        elif f"CMI__{cmi_col}" in rewards_ts.columns:
            canonical_cmi = f"CMI__{cmi_col}"
        elif cmi_col.startswith("CMI__") and cmi_col in rewards_ts.columns:
            canonical_cmi = cmi_col

    # Save base tables
    rewards_ts.to_csv(args.out_ts, index=False)
    trends.to_csv(args.out_trends, index=False)

    # -----------------------------
    # Features + quality
    # -----------------------------
    Q = trends[["r2_Intensity","r2_Fill","r2_Uniformity"]].mean(axis=1, skipna=True)
    Q = Q * np.minimum(trends["n_periods_used"]/5.0, 1.0)
    Q = Q.fillna(0.0).clip(0.0, 1.0)

    feats_df, feat_cols = build_features(trends, feature_set=args.feature_set)
    feats_df["Q"] = Q

    # 1D override?
    feature_one = args.feature_one

    # Phase 1: training subset
    train_mask = (feats_df["Q"] >= args.min_q) & (trends["n_periods_used"] >= args.assign_min_periods)
    feats_df, feat_cols = apply_shrink(feats_df, feat_cols, trends, args.shrink_by_r2, args.shrink_level, feature_one)

    if feature_one is not None:
        mapping = {"intensity_slope": "z_slope_Intensity","fill_slope": "z_slope_Fill","uniformity_slope": "z_slope_Uniformity"}
        feat_cols = [mapping[feature_one]]
    X_train = feats_df.loc[train_mask, feat_cols].values

    ks = [k for k in args.ks if X_train.shape[0] > k]
    if not ks:
        raise ValueError("Not enough eligible samples to evaluate any K. Lower --min-q or add more data.")

    metrics_rows = []
    if args.k_select == "silhouette":
        chosen_k, sils = choose_k_by_silhouette(X_train, args.cluster_method, ks)
        for k in ks: metrics_rows.append({"K": k, "silhouette": sils.get(k, np.nan)})
    else:
        if args.cluster_method != "gmm":
            warnings.warn("BIC selection requires --cluster-method gmm; switching to gmm.", RuntimeWarning)
            args.cluster_method = "gmm"
        chosen_k, bics, aics = choose_k_by_bic_gmm(X_train, ks)
        for k in ks: metrics_rows.append({"K": k, "bic": bics.get(k, np.nan), "aic": aics.get(k, np.nan)})

    if chosen_k is None:
        raise ValueError("Could not select K (no valid silhouette/BIC).")

    model, labels_train, proba_train, centers = fit_cluster_model(X_train, args.cluster_method, chosen_k)
    chosen_sil = silhouette_score(X_train, labels_train) if len(np.unique(labels_train)) > 1 else np.nan

    train_cols = [COL_SEASON, COL_ORCH] + feat_cols + ["Q"]
    assign_train = feats_df.loc[train_mask, train_cols].copy().reset_index(drop=True)
    assign_train["K"] = chosen_k
    assign_train["cluster"] = labels_train
    if proba_train is not None:
        assign_train["best_soft"] = proba_train.max(axis=1)
        for c in range(chosen_k):
            assign_train[f"p_{c}"] = proba_train[:, c]
        assign_train["uncertain"] = (assign_train["best_soft"] < args.confidence_threshold).astype(int)
    else:
        cl, dmin, d2 = assign_by_centroids(assign_train[feat_cols].values, centers)
        assign_train["best_soft"] = 1 - (dmin / (dmin + d2))
        assign_train["Dmin"] = dmin
        assign_train["D2"] = d2
        assign_train["uncertain"] = ((assign_train["best_soft"] < args.confidence_threshold) | (assign_train["Dmin"] > args.distance_threshold)).astype(int)

    label_map = None
    if set(feat_cols) == {"z_slope_Intensity","z_level_Intensity_t5"} and chosen_k in (2,3):
        label_map = label_clusters_cover2(centers, schema=args.cover_label_schema)
        assign_train["label"] = assign_train["cluster"].map(lambda k: label_map.get(int(k)))

    cents = pd.DataFrame(centers, columns=feat_cols)
    cents.insert(0, "cluster", np.arange(chosen_k))

    metrics = pd.DataFrame(metrics_rows)
    metrics["chosen_k"] = chosen_k
    metrics["method"] = args.cluster_method
    fs_used = args.feature_set if feature_one is None else f"one:{feature_one}"
    metrics["feature_set"] = fs_used
    metrics["k_select"] = args.k_select
    metrics["chosen_silhouette"] = chosen_sil
    metrics["shrink_by_r2"] = args.shrink_by_r2
    metrics["shrink_level"] = bool(args.shrink_level)

    # Season silhouettes (train subset)
    season_rows = []
    train_idx = feats_df.index[train_mask].to_numpy()
    for season, idxs in feats_df.loc[train_mask, [COL_SEASON]].reset_index().groupby(COL_SEASON)["index"]:
        idxs = idxs.to_numpy()
        pos = np.searchsorted(train_idx, idxs)
        pos = pos[(pos >= 0) & (pos < len(labels_train))]
        if len(pos) > chosen_k and len(np.unique(labels_train[pos])) > 1:
            sil = silhouette_score(X_train[pos, :], labels_train[pos])
        else:
            sil = np.nan
        season_rows.append({"season": season, "silhouette": sil, "n": int(len(pos))})

    profiles = compute_cluster_profiles(assign_train[[COL_SEASON,COL_ORCH]+feat_cols+["cluster","uncertain"]].copy(),
                                        feat_cols, trends, labels_train, X_train)
    if label_map:
        profiles["label"] = profiles["cluster"].map(lambda k: label_map.get(int(k)))

    # -----------------------------
    # Phase 2: assign ALL feasible rows
    # -----------------------------
    all_mask = (trends["n_periods_used"] >= args.assign_min_periods) & (feats_df["Q"] >= args.assign_min_q)
    X_all = feats_df.loc[all_mask, feat_cols].values

    if args.cluster_method == "kmeans":
        cl_all, dmin_all, d2_all = assign_by_centroids(X_all, centers)
        best_soft_all = 1 - (dmin_all / (dmin_all + d2_all))
        uncertain_all = ((best_soft_all < args.confidence_threshold) | (dmin_all > args.distance_threshold)).astype(int)
    else:
        has_all = np.all(np.isfinite(X_all), axis=1)
        cl_all = np.full(len(X_all), -1, dtype=int)
        best_soft_all = np.full(len(X_all), np.nan)
        uncertain_all = np.ones(len(X_all), dtype=int)
        if has_all.any():
            proba = model.predict_proba(X_all[has_all])
            cl_all[has_all] = np.argmax(proba, axis=1)
            best_soft_all[has_all] = proba.max(axis=1)
            uncertain_all[has_all] = (best_soft_all[has_all] < args.confidence_threshold).astype(int)
        if (~has_all).any():
            cl_f, dmin_f, d2_f = assign_by_centroids(X_all[~has_all], centers)
            cl_all[~has_all] = cl_f
            bs = 1 - (dmin_f / (dmin_f + d2_f))
            best_soft_all[~has_all] = bs
            uncertain_all[~has_all] = ((bs < args.confidence_threshold) | (dmin_f > args.distance_threshold)).astype(int)

    assign_all = feats_df.loc[all_mask, [COL_SEASON, COL_ORCH] + feat_cols + ["Q"]].copy().reset_index(drop=True)
    assign_all["K"] = chosen_k
    assign_all["cluster"] = cl_all
    assign_all["best_soft"] = best_soft_all
    assign_all["uncertain"] = uncertain_all
    if label_map:
        assign_all["label"] = assign_all["cluster"].map(lambda k: label_map.get(int(k)))

    if canonical_cmi:
        assign_train["CMI"] = trends.loc[train_mask, canonical_cmi].values
        assign_all["CMI"] = trends.loc[all_mask, canonical_cmi].values

    # Coverage stats
    cov_rows = []
    for s, g in trends.groupby(COL_SEASON):
        n_trends = int(g.shape[0])
        n_train = int(train_mask[g.index].sum())
        n_all = int(all_mask[g.index].sum())
        cov_rows.append({"season": int(s), "n_trends": n_trends, "n_train": n_train, "n_assigned_all": n_all,
                         "lost_pct_train_only": round(100*(1 - n_train/max(n_trends,1)), 1),
                         "lost_pct_after_all": round(100*(1 - n_all/max(n_trends,1)), 1)})
    coverage = pd.DataFrame(cov_rows)

    # Save template (JSON-safe)
    template = {
        "method": args.cluster_method,
        "K": int(chosen_k),
        "feature_set": fs_used,
        "feature_cols": feat_cols,
        "centers": np.asarray(centers).tolist(),
        "label_map": {str(int(k)): v for k, v in (label_map or {}).items()},
        "shrink_by_r2": args.shrink_by_r2,
        "shrink_level": bool(args.shrink_level),
        "min_q_train": float(args.min_q),
        "assign_min_q": float(args.assign_min_q),
        "assign_min_periods": int(args.assign_min_periods),
        "confidence_threshold": float(args.confidence_threshold),
        "distance_threshold": float(args.distance_threshold),
        "periods": (args.periods if args.periods else DEFAULT_TREND_PERIODS),
    }

    # -----------------------------
    # CMI analysis & clustering (optional)
    # -----------------------------
    cmi_assign = pd.DataFrame(); cmi_cents = pd.DataFrame(); cmi_metrics = pd.DataFrame()
    if canonical_cmi and args.cmi_cluster:
        tmp = trends.rename(columns={canonical_cmi: "CMI_season"})
        cmi_assign, cmi_cents, cmi_metrics = cmi_cluster(tmp, cmi_col="CMI_season",
                                                         method=args.cmi_method,
                                                         kselect=args.cmi_kselect, ks=args.cmi_ks)

    # -----------------------------
    # Flips & transitions (on ALL assignments)
    # -----------------------------
    flips_rows = []
    trans_rows = []
    def feat_dist(a, b):
        a = np.asarray(a, float); b = np.asarray(b, float)
        m = np.isfinite(a) & np.isfinite(b)
        if not m.any(): return np.nan
        return float(np.linalg.norm(a[m] - b[m]) * np.sqrt(len(a) / m.sum()))
    for orch, g in assign_all.groupby(COL_ORCH):
        g = g.copy()
        g["season_sort"] = g[COL_SEASON].apply(season_sort_from_key).astype(int)
        g = g.sort_values("season_sort")
        prev = None
        for _, row in g.iterrows():
            if prev is not None:
                flip = int(row["cluster"] != prev["cluster"])
                dist = feat_dist([row.get(c) for c in feat_cols], [prev.get(c) for c in feat_cols])
                flips_rows.append({
                    "orch_id": orch,
                    "season_from": int(prev["season"]) if str(prev["season"]).isdigit() else prev["season"],
                    "season_to": int(row["season"]) if str(row["season"]).isdigit() else row["season"],
                    "flip": flip,
                    "magnitude": "Large" if (np.isfinite(dist) and dist >= 1.25) else ("Small" if (np.isfinite(dist) and dist < 0.75) else "Medium"),
                    "distance": dist
                })
                trans_rows.append({
                    "season": row["season"],
                    "from_cluster": int(prev["cluster"]),
                    "to_cluster": int(row["cluster"]),
                    "n": 1
                })
            prev = row

    orchard_flips = pd.DataFrame(flips_rows)
    trans_df = pd.DataFrame(trans_rows)
    if not trans_df.empty:
        transitions = trans_df.groupby(["season","from_cluster","to_cluster"])["n"].sum().reset_index()
    else:
        transitions = pd.DataFrame(columns=["season","from_cluster","to_cluster","n"])

    # -----------------------------
    # Write outputs
    # -----------------------------
    assign_train.to_csv(args.out_assign, index=False)
    assign_all.to_csv(args.out_assign_all, index=False)
    cents.to_csv(args.out_centroids, index=False)
    metrics.to_csv(args.out_metrics, index=False)
    pd.DataFrame(season_rows).to_csv(args.out_season_sil, index=False)
    profiles.to_csv(args.out_profiles, index=False)
    coverage.to_csv(args.out_coverage, index=False)
    with open(args.out_template, "w") as f:
        json.dump(_json_safe(template), f, indent=2)
    orchard_flips.to_csv(args.out_flips, index=False)
    transitions.to_csv(args.out_transitions, index=False)

    # CMI-related outputs
    if not cmi_assign.empty and "label" in assign_all.columns and "cmi_label" in cmi_assign.columns:
        cross = assign_all.merge(cmi_assign[[COL_SEASON, COL_ORCH, "cmi_label"]], on=[COL_SEASON, COL_ORCH], how="left")
        ct = cross.pivot_table(index="label", columns="cmi_label", values=COL_ORCH, aggfunc="count", fill_value=0)
        ct_pct = ct.div(ct.sum(axis=1), axis=0).add_suffix("_pct")
        crosstab = pd.concat([ct, ct_pct], axis=1).reset_index()
    else:
        crosstab = pd.DataFrame()

    if canonical_cmi:
        def compute_cover_score(df):
            if "z_slope_Intensity" in df.columns and "z_level_Intensity_t5" in df.columns:
                return df["z_slope_Intensity"].values + 0.8*df["z_level_Intensity_t5"].values
            return None
        cover_score = compute_cover_score(assign_all)
        ca_rows = []
        def block(name, gdf):
            row = {"scope": name, "n": int(len(gdf))}
            if "CMI" in gdf.columns:
                cmi = gdf["CMI"].values
                if cover_score is not None:
                    row["corr_CMI_coverScore"] = float(np.corrcoef(cmi, cover_score[gdf.index])[0,1])
                try:
                    labs = gdf["label"].astype(str).str.lower()
                    y = (labs.str.contains("cover")).astype(int).values
                    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                    lr = LogisticRegression(max_iter=1000)
                    proba = cross_val_predict(lr, cmi.reshape(-1,1), y, cv=skf, method="predict_proba")[:,1]
                    row["auc_cmi_only"] = float(roc_auc_score(y, proba))
                except Exception:
                    row["auc_cmi_only"] = np.nan
            return row
        ca_rows.append(block("GLOBAL", assign_all))
        for s, g in assign_all.groupby(COL_SEASON):
            ca_rows.append({**block(f"SEASON_{s}", g), COL_SEASON: s})
        cmi_analysis_df = pd.DataFrame(ca_rows)
    else:
        cmi_analysis_df = pd.DataFrame(columns=["scope"])

    cmi_analysis_df.to_csv(args.out_cmi_analysis, index=False)
    if not cmi_assign.empty: cmi_assign.to_csv(args.out_cmi_assign, index=False)
    if not cmi_cents.empty:  cmi_cents.to_csv(args.out_cmi_centroids, index=False)
    if not cmi_metrics.empty: cmi_metrics.to_csv(args.out_cmi_metrics, index=False)
    if not crosstab.empty:    crosstab.to_csv(args.out_crosstab, index=False)

    # Orchard profiles from ALL assignments
    orchard_profiles = pd.DataFrame()
    if "label" in assign_all.columns:
        rows = []
        for orch, g in assign_all.groupby(COL_ORCH):
            g = g.sort_values(COL_SEASON, key=lambda s: s.map(season_sort_from_key))
            lab_counts = g["label"].value_counts(dropna=True)
            majority = lab_counts.idxmax() if not lab_counts.empty else ""
            flip_count = int((g["cluster"] != g["cluster"].shift(1)).sum())
            rows.append({COL_ORCH: orch, "n_seasons": int(len(g)), "majority_cover_label": majority,
                         "flip_count": flip_count, "uncertain_rate": float(np.mean(g["uncertain"]))})
        orchard_profiles = pd.DataFrame(rows)
    orchard_profiles.to_csv(args.out_orchard_profiles, index=False)

    # -----------------------------
    # Build and write viz table
    # -----------------------------
    print(f"Building viz_table with {len(orchard_flips) if not orchard_flips.empty else 0} flip records...")
    viz = build_viz_table(assign_all, trends, flips_df=orchard_flips, cmi_assign_df=cmi_assign,
                          season_mid_mmdd=args.season_mid_mmdd)
    viz.to_csv(args.out_viz, index=False)

    # --- Production-ready outputs ---
    emit_production_outputs(
        viz,
        out_attrs_dir=args.out_attrs_dir,
        out_streaks=args.out_streaks,
        out_manifest=args.out_manifest,
        ui_uncertain_threshold=args.ui_uncertain_threshold
    )
    print("Done. Wrote all standard outputs + viz_table.csv.")

# ---------------- Production Outputs: Per-season attrs, Streaks, Manifest ----------------
def _confidence_bucket(bsn: "pd.Series") -> "pd.Series":
    try:
        bins  = [-1.0, 0.15, 0.35, 0.65, 0.85, 2.0]
        labels = ["Coin Toss", "Low", "Average", "Good", "High"]
        return pd.cut(bsn.fillna(0.0), bins=bins, labels=labels, include_lowest=True).astype(str)
    except Exception:
        # Fallback if pandas is not available in type context
        import pandas as _pd
        return _pd.cut(bsn.fillna(0.0), bins=bins, labels=labels, include_lowest=True).astype(str)

def emit_production_outputs(viz: "pd.DataFrame", out_attrs_dir: str, out_streaks: str,
                            out_manifest: str, ui_uncertain_threshold: float) -> None:
    import os, json
    import numpy as np
    import pandas as pd

    os.makedirs(out_attrs_dir, exist_ok=True)

    # Seasons ordered by season_sort (ints)
    seasons = sorted(viz["season_sort"].dropna().astype(int).unique().tolist())

    # ---------- Per-season attribute shards ----------
    for s in seasons:
        sub = viz.loc[viz["season_sort"] == s].copy()

        # Two-class mapping for production
        sub["cluster_role_mapped"] = np.where(sub["cluster_role"] == "Cover Crops", "Cover", "Bare")

        # UI "uncertain" rule based on best_soft_norm (not the model's q/threshold)
        sub["uncertain"] = sub["best_soft_norm"].fillna(0.0) <= float(ui_uncertain_threshold)

        # Flip convenience boolean
        tsly = sub["time_since_last_flip"]
        sub["flipped_this_season"] = tsly.notna() & (tsly.astype(float) == 0.0)

        # Optional UI hover label
        sub["confidence_bucket"] = _confidence_bucket(sub["best_soft_norm"])

        keep = [
            "orch_id", "cluster_role_mapped", "best_soft_norm", "uncertain",
            "flipped_this_season", "streak_len_current", "time_since_last_flip",
            "npts_total", "env_z_season", "stratum",
        ]
        # Only keep columns that exist (for safety across variants)
        keep_existing = [c for c in keep if c in sub.columns]
        sub = sub[keep_existing].copy()

        out_path = os.path.join(out_attrs_dir, f"{s}.csv")
        sub.to_csv(out_path, index=False)

    # ---------- Time-agnostic streaks ----------
    # Confident cover = 'Cover Crops' AND best_soft_norm > threshold
    ok = (viz["cluster_role"] == "Cover Crops") & (viz["best_soft_norm"].fillna(0.0) > float(ui_uncertain_threshold))
    cc = viz.loc[ok, ["orch_id", "season_sort"]].dropna().copy()
    if not cc.empty:
        cc["season_sort"] = cc["season_sort"].astype(int)

        # Build a pivot of 0/1 flags per orchard × season
        flags = (
            cc.assign(flag=1)
              .pivot_table(index="orch_id", columns="season_sort", values="flag", fill_value=0, aggfunc="max")
              .reindex(columns=seasons, fill_value=0)
        )
    else:
        import pandas as pd
        flags = pd.DataFrame(index=viz["orch_id"].dropna().unique())
        for s in seasons:
            flags[s] = 0

    def longest_run(arr):
        best = run = 0
        for v in arr:
            run = run + 1 if v else 0
            if run > best:
                best = run
        return best

    masks = []
    longs = []
    years_active = flags.sum(axis=1).astype(int).tolist()
    for row in flags.itertuples(index=False, name=None):
        m = 0
        for i, v in enumerate(row):
            if int(v) == 1:
                m |= (1 << i)
        masks.append(m)
        longs.append(longest_run(row))

    import pandas as pd
    streaks = pd.DataFrame({
        "orch_id": flags.index,
        "mask": masks,
        "longest": longs,
        "years_active": years_active,
    })
    streaks["ever_cover"]  = streaks["years_active"] > 0
    streaks["never_cover"] = streaks["years_active"] == 0
    streaks.to_csv(out_streaks, index=False)

    # ---------- Manifest ----------
    manifest = {
        "schema_version": 1,
        "years": seasons,
        "ui_uncertain_threshold": float(ui_uncertain_threshold),
        "bit_order": "earliest=bit0",
    }
    with open(out_manifest, "w") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
