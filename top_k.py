import os, json, warnings, datetime
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.compose import ColumnTransformer

RANDOM_STATE = 42
N_JOBS = -1
N_SPLITS = 5


def load_data(input_path, target_col, pos_label, id_col):
    df = pd.read_csv(input_path)
    y = (df[target_col] == pos_label).astype(int).values
    X = df.drop(columns=[col for col in [id_col, target_col] if col in df.columns])
    return df, X, y, X.columns.tolist()


def detect_binary_columns(X: pd.DataFrame):
    bin_cols = []
    for c in X.columns:
        vals = pd.unique(X[c].dropna())
        if len(vals) == 0:
            continue
        if set(np.unique(vals)).issubset({0, 1}):
            bin_cols.append(c)
    cont_cols = [c for c in X.columns if c not in bin_cols]
    return bin_cols, cont_cols


def univariate_auc_scores_cv(X, y, bin_cols, cont_cols):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    m = X.shape[1]
    acc, cnt = np.zeros(m), np.zeros(m, dtype=int)
    col_index = {c: i for i, c in enumerate(X.columns)}

    for tr, te in skf.split(X, y):
        Xtr, Xte, yte = X.iloc[tr], X.iloc[te], y[te]

        imp_bin = SimpleImputer(strategy="most_frequent")
        imp_con = SimpleImputer(strategy="median")
        Xte_bin = imp_bin.fit(Xtr[bin_cols]).transform(Xte[bin_cols]) if bin_cols else np.zeros((len(te), 0))
        Xte_con = imp_con.fit(Xtr[cont_cols]).transform(Xte[cont_cols]) if cont_cols else np.zeros((len(te), 0))

        Xte_imp = np.zeros((len(te), m))
        if bin_cols:
            for j, c in enumerate(bin_cols):
                Xte_imp[:, col_index[c]] = Xte_bin[:, j]
        if cont_cols:
            for j, c in enumerate(cont_cols):
                Xte_imp[:, col_index[c]] = Xte_con[:, j]

        for c in X.columns:
            j = col_index[c]
            col = Xte_imp[:, j]
            if np.nanstd(col) == 0:
                continue
            try:
                auc = roc_auc_score(yte, col)
                s = abs(auc - 0.5) * 2.0
                acc[j] += s
                cnt[j] += 1
            except Exception:
                pass

    out = np.zeros(m)
    mask = cnt > 0
    out[mask] = acc[mask] / cnt[mask]
    return out


def mutual_info_scores_cv(X, y, bin_cols, cont_cols):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    m = X.shape[1]
    acc, cnt = np.zeros(m), np.zeros(m, dtype=int)
    col_index = {c: i for i, c in enumerate(X.columns)}
    discrete_mask = np.array([c in bin_cols for c in X.columns])

    for tr, te in skf.split(X, y):
        Xtr, Xte, yte = X.iloc[tr], X.iloc[te], y[te]

        imp_bin = SimpleImputer(strategy="most_frequent")
        imp_con = SimpleImputer(strategy="median")
        Xte_bin = imp_bin.fit(Xtr[bin_cols]).transform(Xte[bin_cols]) if bin_cols else np.zeros((len(te), 0))
        Xte_con = imp_con.fit(Xtr[cont_cols]).transform(Xte[cont_cols]) if cont_cols else np.zeros((len(te), 0))

        Xte_imp = np.zeros((len(te), m))
        if bin_cols:
            for j, c in enumerate(bin_cols):
                Xte_imp[:, col_index[c]] = Xte_bin[:, j]
        if cont_cols:
            for j, c in enumerate(cont_cols):
                Xte_imp[:, col_index[c]] = Xte_con[:, j]

        try:
            mi = mutual_info_classif(Xte_imp, yte, random_state=RANDOM_STATE, discrete_features=discrete_mask)
            acc += mi
            cnt += 1
        except Exception:
            pass

    out = np.zeros(m)
    mask = cnt > 0
    out[mask] = acc[mask] / cnt[mask]
    return out


def get_preprocessor_for_lr(bin_cols, cont_cols):
    transformers = []
    if bin_cols:
        transformers.append(("bin", SimpleImputer(strategy="most_frequent"), bin_cols))
    if cont_cols:
        transformers.append(("con", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), cont_cols))
    return ColumnTransformer(transformers, remainder="drop")


def l1_logreg_abs_coefs_cv(X, y, bin_cols, cont_cols):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    m = X.shape[1]
    acc, cnt = np.zeros(m), np.zeros(m, dtype=int)

    for tr, _ in skf.split(X, y):
        Xtr, ytr = X.iloc[tr], y[tr]
        pre = get_preprocessor_for_lr(bin_cols, cont_cols)
        pipe = make_pipeline(
            pre,
            LogisticRegression(
                penalty="l1",
                solver="saga",
                C=1.0,
                max_iter=5000,
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
            ),
        )
        try:
            pipe.fit(Xtr, ytr)
            lr = pipe.named_steps["logisticregression"]
            ordered_cols = bin_cols + cont_cols
            coefs = np.zeros(m)
            coefs[np.array([X.columns.get_loc(c) for c in ordered_cols])] = np.abs(lr.coef_.ravel())
            acc += coefs
            cnt += 1
        except Exception:
            pass

    out = np.zeros(m)
    mask = cnt > 0
    out[mask] = acc[mask] / cnt[mask]
    return out


def rf_importance_cv(X, y, bin_cols, cont_cols):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    m = X.shape[1]
    acc, cnt = np.zeros(m), np.zeros(m, dtype=int)
    col_index = {c: i for i, c in enumerate(X.columns)}

    for tr, _ in skf.split(X, y):
        Xtr, ytr = X.iloc[tr], y[tr]

        imp_bin = SimpleImputer(strategy="most_frequent")
        imp_con = SimpleImputer(strategy="median")
        Xtr_bin = imp_bin.fit_transform(Xtr[bin_cols]) if bin_cols else np.zeros((len(tr), 0))
        Xtr_con = imp_con.fit_transform(Xtr[cont_cols]) if cont_cols else np.zeros((len(tr), 0))

        Xtr_imp = np.zeros((len(tr), m))
        if bin_cols:
            for j, c in enumerate(bin_cols):
                Xtr_imp[:, col_index[c]] = Xtr_bin[:, j]
        if cont_cols:
            for j, c in enumerate(cont_cols):
                Xtr_imp[:, col_index[c]] = Xtr_con[:, j]

        rf = RandomForestClassifier(
            n_estimators=500,
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            class_weight="balanced_subsample",
        )
        try:
            rf.fit(Xtr_imp, ytr)
            acc += rf.feature_importances_
            cnt += 1
        except Exception:
            pass

    out = np.zeros(m)
    mask = cnt > 0
    out[mask] = acc[mask] / cnt[mask]
    return out


def permutation_importance_lr(X, y, bin_cols, cont_cols):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    importances = []
    for tr, te in skf.split(X, y):
        X_tr, y_tr, X_te, y_te = X.iloc[tr], y[tr], X.iloc[te], y[te]

        if np.unique(y_tr).size < 2 or np.unique(y_te).size < 2:
            importances.append(np.zeros(X.shape[1]))
            continue

        pre = get_preprocessor_for_lr(bin_cols, cont_cols)
        pipe = make_pipeline(
            pre,
            LogisticRegression(
                penalty="l2",
                solver="lbfgs",
                max_iter=2000,
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
            ),
        )
        try:
            pipe.fit(X_tr, y_tr)
            pi = permutation_importance(
                pipe,
                X_te,
                y_te,
                scoring="roc_auc",
                n_repeats=10,
                random_state=RANDOM_STATE,
                n_jobs=N_JOBS,
            ).importances_mean
        except Exception:
            pi = np.zeros(X.shape[1])
        importances.append(pi)
    return np.mean(np.vstack(importances), axis=0)


# ---------- utils ----------
def minmax_scale(x):
    x = np.array(x, dtype=float)
    mn, mx = np.nanmin(x), np.nanmax(x)
    if mx <= mn:
        return np.zeros_like(x)
    return (x - mn) / (mx - mn)


def main(input_path, target_col, pos_label, id_col="video_id", top_k=15):
    print("🔎 Wczytuję dane…")
    df, X, y, feature_names = load_data(input_path, target_col, pos_label, id_col)
    bin_cols, cont_cols = detect_binary_columns(X)
    print(f"→ próbki={len(y)}, cechy={X.shape[1]}, binarne={len(bin_cols)}, ciągłe={len(cont_cols)}")

    print("Obliczanie rankingów...")
    s_auc = univariate_auc_scores_cv(X, y, bin_cols, cont_cols)
    s_mi = mutual_info_scores_cv(X, y, bin_cols, cont_cols)
    s_lr = l1_logreg_abs_coefs_cv(X, y, bin_cols, cont_cols)
    s_rf = rf_importance_cv(X, y, bin_cols, cont_cols)
    s_pi = permutation_importance_lr(X, y, bin_cols, cont_cols)

    scores = pd.DataFrame({
        "feature": feature_names,
        "univariate_auc": s_auc,
        "mutual_info": s_mi,
        "logreg_L1_coef": s_lr,
        "rf_importance": s_rf,
        "perm_importance_lr": s_pi,
    })

    for col in ["univariate_auc", "mutual_info", "logreg_L1_coef", "rf_importance", "perm_importance_lr"]:
        scores[col + "_scaled"] = minmax_scale(scores[col])

    scores["meta_score"] = scores[[c for c in scores.columns if c.endswith("_scaled")]].mean(axis=1)
    scores = scores.sort_values("meta_score", ascending=False).reset_index(drop=True)

    out_dir = f"reports/{target_col}"
    os.makedirs(out_dir, exist_ok=True)

    scores.to_csv(f"{out_dir}/feature_ranking.csv", index=False)
    print(f"✅ Zapisano ranking cech: {out_dir}/feature_ranking.csv")
    n_features = scores.shape[0]
    if top_k is None:
        top_k = 0
    try:
        top_k = int(top_k)
    except Exception:
        top_k = 0
    if top_k < 0:
        top_k = 0
    if top_k > n_features:
        top_k = n_features

    scores.head(top_k).to_csv(f"{out_dir}/feature_ranking_top{top_k}.csv", index=False)
    print(f"✅ Zapisano TOP-{top_k} ranking cech: {out_dir}/feature_ranking_top{top_k}.csv")

    reduced = df[[c for c in [id_col, target_col] if c in df.columns] + scores["feature"].head(top_k).tolist()]
    reduced.to_csv(f"{out_dir}/dataset_topK.csv", index=False)
    print(f"✅ Zapisano zredukowany dataset ({top_k} cech): {out_dir}/dataset_topK.csv")

    report = {
        "created": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "target_col": target_col,
        "pos_label": pos_label,
        "n_samples": len(y),
        "n_features": X.shape[1],
        "top_k": top_k,
        "top_features": scores["feature"].head(top_k).tolist(),
        "top_features_with_scores": scores[["feature", "meta_score"]].head(top_k).to_dict(orient="records"),
    }
    with open(f"{out_dir}/feature_screening.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n🏁 TOP {top_k} cech wg meta_score:")
    if top_k > 0:
        print(scores[["feature", "meta_score"]].head(top_k).to_string(index=False))
    else:
        print("(brak wybranych cech)")


if __name__ == "__main__":
    main(
        input_path="/Users/macbook/git/ops-classification/last_change/dataset_oioe_cleaned.csv",
        target_col="OiOe",
        pos_label="Oi",
        id_col="video_id",
        top_k=15,
    )
