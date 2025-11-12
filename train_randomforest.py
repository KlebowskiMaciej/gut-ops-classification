import os, json, random, warnings, joblib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score,
    recall_score, log_loss
)
from sklearn.impute import SimpleImputer

try:
    from imblearn.over_sampling import SMOTE
    HAS_SMOTE = True
except Exception:
    HAS_SMOTE = False

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)
random.seed(RANDOM_STATE)

def load_dataset(input_csv, id_col, target_col, pos_label):
    df = pd.read_csv(input_csv)
    y = (df[target_col] == pos_label).astype(int).values
    X = df.drop(columns=[c for c in [id_col, target_col] if c in df.columns])
    return df, X, y, X.columns.tolist()

def make_builder(X):
    cols = X.columns
    bin_cols, cont_cols = [], []
    for c in cols:
        vals = pd.unique(X[c].dropna())
        if set(vals).issubset({0, 1}) and len(vals) <= 2:
            bin_cols.append(c)
        else:
            cont_cols.append(c)
    bin_imp = SimpleImputer(strategy="most_frequent")
    cont_imp = SimpleImputer(strategy="median")

    def build(X_part, fit=False):
        Xb = X_part[bin_cols].copy() if bin_cols else pd.DataFrame(index=X_part.index)
        Xc = X_part[cont_cols].copy() if cont_cols else pd.DataFrame(index=X_part.index)
        if fit:
            if bin_cols:
                bin_imp.fit(Xb)
            if cont_cols:
                cont_imp.fit(Xc)
        if bin_cols:
            Xb = bin_imp.transform(Xb)
        if cont_cols:
            Xc = cont_imp.transform(Xc)
        if bin_cols and cont_cols:
            Xall = np.hstack([Xb, Xc])
        elif bin_cols:
            Xall = Xb
        else:
            Xall = Xc
        return np.asarray(Xall, dtype=np.float64)

    return build, bin_cols, cont_cols

def cv_metrics(clf, X, y, build_X, cv_splitter):
    out = []
    for fold_id, (tr, te) in enumerate(cv_splitter.split(X, y), 1):
        Xtr = build_X(X.iloc[tr], fit=True)
        Xte = build_X(X.iloc[te], fit=False)
        ytr, yte = y[tr], y[te]

        if HAS_SMOTE:
            try:
                min_class = np.bincount(ytr).min()
                k_neighbors = max(1, min(5, min_class - 1))
                sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
                Xtr, ytr = sm.fit_resample(Xtr, ytr)
            except Exception:
                pass

        clf.fit(Xtr, ytr)
        proba = clf.predict_proba(Xte)[:, 1]
        pred = (proba >= 0.5).astype(int)

        out.append(dict(
            fold=fold_id,
            auc=roc_auc_score(yte, proba),
            f1=f1_score(yte, pred),
            acc=accuracy_score(yte, pred),
            prec=precision_score(yte, pred, zero_division=0),
            rec=recall_score(yte, pred),
            logloss=log_loss(yte, proba, labels=[0, 1])
        ))

    df = pd.DataFrame(out)
    return df

def sample_space(center=None, shrink=1.0):
    base = dict(
        n_estimators=(300, 1500),
        max_depth=(-1, 30), 
        max_features=["sqrt", "log2", 0.5, 0.7, 1.0],
        min_samples_leaf=(1, 20),
        criterion=["gini", "entropy"],
        max_samples=[None, 0.7, 0.8, 0.9, 1.0],
    )
    if center is None:
        return base

    def bound(lo, hi, c):
        width = (hi - lo) * 0.5 * shrink
        lo2, hi2 = max(lo, c - width), min(hi, c + width)
        return (lo, hi) if lo2 >= hi2 else (lo2, hi2)

    return dict(
        n_estimators=bound(*base["n_estimators"], center["n_estimators"]),
        max_depth=bound(*base["max_depth"], center["max_depth"]),
        max_features=base["max_features"],
        min_samples_leaf=bound(*base["min_samples_leaf"], center["min_samples_leaf"]),
        criterion=base["criterion"],
        max_samples=base["max_samples"],
    )


def draw_params(r):
    def pick_num(lo, hi, integer=False):
        v = lo + (hi - lo) * random.random()
        return int(round(v)) if integer else v

    max_depth_choices = [-1] + list(range(1, 31))
    return dict(
        n_estimators=pick_num(*r["n_estimators"], integer=True),
        max_depth=random.choice(max_depth_choices),
        max_features=random.choice(r["max_features"]),
        min_samples_leaf=pick_num(*r["min_samples_leaf"], integer=True),
        criterion=random.choice(r["criterion"]),
        max_samples=random.choice(r["max_samples"]),
    )


def make_rf(p):
    return RandomForestClassifier(
        n_estimators=int(p["n_estimators"]),
        max_depth=None if p["max_depth"] <= 0 else int(p["max_depth"]),
        max_features=p["max_features"],
        min_samples_leaf=int(p["min_samples_leaf"]),
        criterion=p["criterion"],
        max_samples=p["max_samples"],
        class_weight="balanced_subsample",
        oob_score=True,
        bootstrap=True,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

def main(ID_COL, TARGET_COL, POS_LABEL, MODEL_NAME, INPUT_CSV_PATH):
    model_dir = os.path.join("models", TARGET_COL)
    report_dir = os.path.join("reports", TARGET_COL)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    df, X, y, feat = load_dataset(INPUT_CSV_PATH, ID_COL, TARGET_COL, POS_LABEL)
    build_X, bin_cols, cont_cols = make_builder(X)
    print(f"ℹ️  n={len(X)}, features={len(feat)}")

    ranges = sample_space(None)
    best_params, best_score = None, -1.0
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)

    for it in range(1, 4):
        print(f"\n🔁  {it}/3")
        cand = 40 if it == 1 else 25
        shrink = 1.0 if it == 1 else (0.6 if it == 2 else 0.4)
        ranges = sample_space(best_params if best_params else None, shrink)

        for _ in range(cand):
            p = draw_params(ranges)
            rf = make_rf(p)
            df_cv = cv_metrics(rf, X, y, build_X, rskf)
            score = df_cv["auc"].mean()
            if score > best_score:
                best_score, best_params = score, p
        print(f" → best AUC: {best_score:.4f}")

    print("\n✅ Best params:", best_params)

    # final CV
    best_rf = make_rf(best_params)
    df_cv = cv_metrics(best_rf, X, y, build_X, rskf)
    mean = df_cv.mean().to_dict()
    std = df_cv.std(ddof=1).to_dict()

    print("📊 CV:", {k: f"{mean[k]:.4f}±{std[k]:.4f}" for k in mean})

    X_all = build_X(X, fit=True)
    y_all = y
    if HAS_SMOTE:
        try:
            min_class = np.bincount(y_all).min()
            k_neighbors = max(1, min(5, min_class - 1))
            sm = SMOTE(random_state=RANDOM_STATE, k_neighbors=k_neighbors)
            X_all, y_all = sm.fit_resample(X_all, y_all)
        except Exception:
            pass

    best_rf.fit(X_all, y_all)

    model_path = os.path.join(model_dir, f"randomforest_{TARGET_COL}.pkl")
    report_path = os.path.join(report_dir, f"randomforest_{TARGET_COL}.json")

    joblib.dump({"model": best_rf, "features": feat, "bin": bin_cols, "cont": cont_cols}, model_path)

    with open(report_path, "w") as f:
        json.dump(
            dict(
                model=MODEL_NAME,
                target=TARGET_COL,
                params=best_params,
                best_auc=best_score,
                cv_folds=df_cv.to_dict(orient="records"),
                cv_mean=mean,
                cv_std=std,
                features=feat,
                oob_score=best_rf.oob_score_ if hasattr(best_rf, "oob_score_") else None,
                timestamp=datetime.now().isoformat(),
            ),
            f,
            indent=2,
        )



if __name__ == "__main__":
    main(
        ID_COL="video_id",
        TARGET_COL="ObserverDecider",
        POS_LABEL="Observer",
        MODEL_NAME="randomforest",
        INPUT_CSV_PATH="observer_decider_dataset_cleaned.csv",
    )
