
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
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

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
    scaler = StandardScaler()

    def build(X_part, fit=False):
        Xb = X_part[bin_cols].copy() if bin_cols else pd.DataFrame(index=X_part.index)
        Xc = X_part[cont_cols].copy() if cont_cols else pd.DataFrame(index=X_part.index)
        if fit:
            if bin_cols:
                bin_imp.fit(Xb)
            if cont_cols:
                cont_imp.fit(Xc)
                scaler.fit(cont_imp.transform(Xc))
        if bin_cols:
            Xb = bin_imp.transform(Xb)
        if cont_cols:
            Xc = cont_imp.transform(Xc)
            Xc = scaler.transform(Xc)
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


def make_lr(params):
    penalty = params["penalty"]
    solver = "saga" if penalty in ["l1", "elasticnet"] else "lbfgs"
    return LogisticRegression(
        penalty=penalty,
        solver=solver,
        C=float(params["C"]),
        l1_ratio=params.get("l1_ratio", None),
        max_iter=5000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced"
    )

def sample_space(center=None, shrink=1.0):
    base = dict(
        penalty=["l2", "l1", "elasticnet"],
        C=(1e-4, 1e4),
        l1_ratio=(0.0, 1.0),
    )
    if center is None:
        return base

    def narrow_log(lo, hi, c):
        import math
        width = (math.log10(hi) - math.log10(lo)) * 0.5 * shrink
        lc = math.log10(c)
        nlo = max(math.log10(lo), lc - width)
        nhi = min(math.log10(hi), lc + width)
        if nlo >= nhi:
            return (lo, hi)
        return (10**nlo, 10**nhi)

    return dict(
        penalty=base["penalty"],
        C=narrow_log(*base["C"], center["C"]),
        l1_ratio=(0.0, 1.0),
    )


def draw_params(r):
    import math
    lo, hi = r["C"]
    C = 10**(math.log10(lo) + (math.log10(hi) - math.log10(lo)) * random.random())
    penalty = random.choice(r["penalty"])
    l1_ratio = None
    if penalty == "elasticnet":
        l1_ratio = random.random()
    return {"penalty": penalty, "C": C, "l1_ratio": l1_ratio}

def main(ID_COL, TARGET_COL, POS_LABEL, MODEL_NAME, INPUT_CSV_PATH):
    model_dir = os.path.join("models", TARGET_COL)
    report_dir = os.path.join("reports", TARGET_COL)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    df, X, y, feat = load_dataset(INPUT_CSV_PATH, ID_COL, TARGET_COL, POS_LABEL)
    build_X, bin_cols, cont_cols = make_builder(X)
    print(f"ℹ️  n={len(X)}, features={len(feat)}")

    best_params, best_score = None, -1.0
    ranges = sample_space()
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)

    for it in range(1, 4):
        print(f"\n🔁  {it}/3")
        cand = 40 if it == 1 else 30
        shrink = 1.0 if it == 1 else (0.6 if it == 2 else 0.4)
        ranges = sample_space(best_params if best_params else None, shrink)

        for _ in range(cand):
            params = draw_params(ranges)
            clf = make_lr(params)
            df_cv = cv_metrics(clf, X, y, build_X, rskf)
            score = df_cv["auc"].mean()
            if score > best_score:
                best_score, best_params = score, params
        print(f" → best AUC: {best_score:.4f}")

    print("\n✅ Best params:", best_params)

    best_clf = make_lr(best_params)
    df_cv = cv_metrics(best_clf, X, y, build_X, rskf)
    mean = df_cv.mean().to_dict()
    std = df_cv.std(ddof=1).to_dict()

    print("📊 CV:", {k: f"{mean[k]:.4f}±{std[k]:.4f}" for k in mean})

    # retrain
    X_all = build_X(X, fit=True)
    y_all = y
    best_clf.fit(X_all, y_all)

    model_path = os.path.join(model_dir, f"logreg_{TARGET_COL}.pkl")
    report_path = os.path.join(report_dir, f"logreg_{TARGET_COL}.json")

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
        TARGET_COL="ObserverDecider",
        POS_LABEL="Observer",
        MODEL_NAME="logreg",
        INPUT_CSV_PATH="observer_decider_dataset_cleaned.csv",
    )
