import os, json, random, warnings, joblib
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score, precision_score,
    recall_score, log_loss
)
from sklearn.impute import SimpleImputer

try:
    from lightgbm import LGBMClassifier
except Exception:
    LGBMClassifier = None

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
        if len(vals) <= 2 and set(vals).issubset({0, 1}):
            bin_cols.append(c)
        else:
            cont_cols.append(c)
    bin_imp = SimpleImputer(strategy="most_frequent")
    cont_imp = SimpleImputer(strategy="median")

    def build(X_part, fit=False):
        Xb = X_part[bin_cols].copy() if bin_cols else pd.DataFrame(index=X_part.index)
        Xc = X_part[cont_cols].copy() if cont_cols else pd.DataFrame(index=X_part.index)
        if fit:
            if bin_cols: bin_imp.fit(Xb)
            if cont_cols: cont_imp.fit(Xc)
        if bin_cols: Xb = bin_imp.transform(Xb)
        if cont_cols: Xc = cont_imp.transform(Xc)
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

    return pd.DataFrame(out)


def make_lgbm(params):
    n_estimators = int(params["n_estimators"])
    num_leaves = max(2, int(params.get("num_leaves", 31)))
    learning_rate = float(params["learning_rate"])
    md = int(params.get("max_depth", -1))
    max_depth = md if md != 0 else -1
    return LGBMClassifier(
        random_state=RANDOM_STATE,
        n_estimators=n_estimators,
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=float(params["subsample"]),
        colsample_bytree=float(params["colsample_bytree"]),
        min_child_samples=int(params["min_child_samples"]),
        reg_alpha=float(params["reg_alpha"]),
        reg_lambda=float(params["reg_lambda"]),
        class_weight="balanced",
        n_jobs=-1,
        max_bin=63,
        min_data_in_bin=1,
        min_split_gain=0.0,
        force_col_wise=True,
        verbose=-1
    )


def sample_space(n_rows, center=None, shrink=1.0):
    max_leaves = max(8, min(64, n_rows // 2))
    base = dict(
        n_estimators=(150, 800),
        num_leaves=(8, max_leaves),
        learning_rate=(0.02, 0.2),
        max_depth=(-1, 8),
        subsample=(0.7, 1.0),
        colsample_bytree=(0.7, 1.0),
        min_child_samples=(2, max(5, n_rows // 4)),
        reg_alpha=(0.0, 1.0),
        reg_lambda=(0.0, 2.0),
    )
    if center is None:
        return base

    narrowed = {}
    for k, (a, b) in base.items():
        c = center[k]
        if k == "max_depth" and c == -1:
            narrowed[k] = (-1, 6)
            continue
        width = (b - a) * 0.5 * shrink
        lo, hi = max(a, c - width), min(b, c + width)
        if lo >= hi: lo, hi = a, b
        narrowed[k] = (lo, hi)
    return narrowed


def draw_params(ranges):
    def pick(lo, hi, integer=False):
        if lo == -1 and hi != -1:
            if random.random() < 0.2:
                return -1
            lo = 3
        v = lo + (hi - lo) * random.random()
        return int(round(v)) if integer else v

    return {
        "n_estimators": pick(*ranges["n_estimators"], integer=True),
        "num_leaves": max(4, pick(*ranges["num_leaves"], integer=True)),
        "learning_rate": pick(*ranges["learning_rate"]),
        "max_depth": pick(*ranges["max_depth"], integer=True),
        "subsample": pick(*ranges["subsample"]),
        "colsample_bytree": pick(*ranges["colsample_bytree"]),
        "min_child_samples": max(2, pick(*ranges["min_child_samples"], integer=True)),
        "reg_alpha": pick(*ranges["reg_alpha"]),
        "reg_lambda": pick(*ranges["reg_lambda"]),
    }


def fallback_params(p):
    q = dict(p)
    q["min_child_samples"] = max(2, int(round(p["min_child_samples"] * 0.5)))
    q["num_leaves"] = max(p["num_leaves"], 31)
    if p["max_depth"] != -1:
        q["max_depth"] = max(4, int(p["max_depth"]))
    q["learning_rate"] = max(0.01, p["learning_rate"] * 0.7)
    return q

def main(ID_COL, TARGET_COL, POS_LABEL, MODEL_NAME, INPUT_CSV_PATH):
    if LGBMClassifier is None:
        return

    model_dir = os.path.join("models", TARGET_COL)
    report_dir = os.path.join("reports", TARGET_COL)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    df, X, y, feat = load_dataset(INPUT_CSV_PATH, ID_COL, TARGET_COL, POS_LABEL)
    build_X, bin_cols, cont_cols = make_builder(X)

    n_rows = len(X)
    best_params, best_score = None, -1.0
    ranges = sample_space(n_rows)
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)

    for it in range(1, 4):
        print(f"\n🔁 iter {it}/3")
        cand = 24 if it == 1 else 16
        shrink = 1.0 if it == 1 else (0.6 if it == 2 else 0.4)
        ranges = sample_space(n_rows, best_params if best_params else None, shrink)

        for _ in range(cand):
            params = draw_params(ranges)
            if params["max_depth"] != -1:
                params["num_leaves"] = int(min(params["num_leaves"], 2 ** params["max_depth"]))

            clf = make_lgbm(params)
            df_cv = cv_metrics(clf, X, y, build_X, rskf)
            score = df_cv["auc"].mean()

            if score < 0.55:
                p2 = fallback_params(params)
                clf2 = make_lgbm(p2)
                df_cv2 = cv_metrics(clf2, X, y, build_X, rskf)
                if df_cv2["auc"].mean() > score:
                    params, df_cv, score = p2, df_cv2, df_cv2["auc"].mean()

            if score > best_score:
                best_score, best_params = score, params

        print(f" → best AUC: {best_score:.4f}")

    print("\n✅ Best params:", best_params)

    best_clf = make_lgbm(best_params)
    df_cv = cv_metrics(best_clf, X, y, build_X, rskf)
    mean, std = df_cv.mean().to_dict(), df_cv.std(ddof=1).to_dict()

    print("📊 CV:", {k: f"{mean[k]:.4f}±{std[k]:.4f}" for k in mean})

    X_all = build_X(X, fit=True)
    best_clf.fit(X_all, y)

    model_path = os.path.join(model_dir, f"lightgbm_{TARGET_COL}.pkl")
    report_path = os.path.join(report_dir, f"lightgbm_{TARGET_COL}.json")

    joblib.dump({"model": best_clf, "features": feat,
                 "bin": bin_cols, "cont": cont_cols}, model_path)

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
                timestamp=datetime.now().isoformat(),
            ),
            f,
            indent=2,
        )



if __name__ == "__main__":
    main(
        ID_COL="video_id",
        TARGET_COL="OiOe",
        POS_LABEL="Oi",
        MODEL_NAME="lightgbm",
        INPUT_CSV_PATH="dataset_oioe_cleaned.csv",
    )
