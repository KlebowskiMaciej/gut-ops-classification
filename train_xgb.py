
import os, json, random, warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

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

try:
    from xgboost import XGBClassifier
except Exception:
    XGBClassifier = None


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

def sample_space(center=None, n_rows=100, shrink=1.0):
    base = dict(
        n_estimators=(100, 800),
        max_depth=(3, 8),
        learning_rate=(0.01, 0.3),
        subsample=(0.6, 1.0),
        colsample_bytree=(0.5, 1.0),
        gamma=(0.0, 3.0),
        min_child_weight=(1, max(2, n_rows // 20)),
        reg_alpha=(0.0, 1.0),
        reg_lambda=(0.0, 2.0),
    )
    if center is None:
        return base
    narrowed = {}
    for k, (a, b) in base.items():
        c = center[k]
        width = (b - a) * 0.4 * shrink
        lo, hi = max(a, c - width), min(b, c + width)
        if lo >= hi:
            lo, hi = a, b
        narrowed[k] = (lo, hi)
    return narrowed


def draw_params(ranges):
    def rnd(lo, hi, integer=False):
        v = lo + (hi - lo) * random.random()
        return int(round(v)) if integer else float(v)

    return dict(
        n_estimators=rnd(*ranges["n_estimators"], integer=True),
        max_depth=rnd(*ranges["max_depth"], integer=True),
        learning_rate=rnd(*ranges["learning_rate"]),
        subsample=rnd(*ranges["subsample"]),
        colsample_bytree=rnd(*ranges["colsample_bytree"]),
        gamma=rnd(*ranges["gamma"]),
        min_child_weight=rnd(*ranges["min_child_weight"], integer=True),
        reg_alpha=rnd(*ranges["reg_alpha"]),
        reg_lambda=rnd(*ranges["reg_lambda"]),
    )

def main(ID_COL, TARGET_COL, POS_LABEL, MODEL_NAME, INPUT_CSV_PATH):
    if XGBClassifier is None:
        return

    model_dir = os.path.join("models", TARGET_COL)
    report_dir = os.path.join("reports", TARGET_COL)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    df, X, y, feat = load_dataset(INPUT_CSV_PATH, ID_COL, TARGET_COL, POS_LABEL)
    build_X, bin_cols, cont_cols = make_builder(X)
    print(f"ℹ️  n={len(X)}, features={len(feat)}")

    n_rows = len(X)
    ranges = sample_space(None, n_rows)
    best_params, best_score = None, -1.0
    rskf = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=RANDOM_STATE)

    for it in range(1, 4):
        print(f"\n🔁  {it}/3")
        cand = 50 if it == 1 else 30
        shrink = 1.0 if it == 1 else (0.6 if it == 2 else 0.4)
        ranges = sample_space(best_params if best_params else None, n_rows, shrink)

        for _ in range(cand):
            p = draw_params(ranges)
            clf = XGBClassifier(
                use_label_encoder=False,
                objective="binary:logistic",
                random_state=RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
                **p,
            )
            df_cv = cv_metrics(clf, X, y, build_X, rskf)
            score = df_cv["auc"].mean()
            if score > best_score:
                best_score, best_params = score, p
        print(f" → best AUC now: {best_score:.4f}")

    print("\n✅ Best params:", best_params)

    best_clf = XGBClassifier(
        use_label_encoder=False,
        objective="binary:logistic",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbosity=0,
        **best_params,
    )
    df_cv = cv_metrics(best_clf, X, y, build_X, rskf)
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

    best_clf.fit(X_all, y_all)

    import pickle
    model_path = os.path.join(model_dir, f"xgb_{TARGET_COL}.pkl")
    report_path = os.path.join(report_dir, f"xgb_{TARGET_COL}.json")

    with open(model_path, "wb") as f:
        pickle.dump(
            {"model": best_clf, "features": feat, "bin": bin_cols, "cont": cont_cols}, f
        )
    with open(report_path, "w") as f:
        json.dump(
            dict(
                model=MODEL_NAME,
                target=TARGET_COL,
                params=best_params,
                cv_folds=df_cv.to_dict(orient="records"),
                cv_mean=mean,
                cv_std=std,
                features=feat,
            ),
            f,
            indent=2,
        )


if __name__ == "__main__":
    main(
        ID_COL="video_id",
        TARGET_COL="ObserverDecider",
        POS_LABEL="Observer",
        MODEL_NAME="xgboost",
        INPUT_CSV_PATH="observer_decider_dataset_cleaned.csv",
    )
