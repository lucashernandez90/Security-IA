"""
train.py – Treina XGBoost e LightGBM usando os .npy de data/processed.
Exemplo:
    python src/train.py --in-dir data/processed --out-models models --out-reports reports --task binary
    
"""
from __future__ import annotations
import argparse
import json
import os

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report

from utils import ensure_dir, save_json, compute_metrics, plot_confusion_matrix


def load_processed(in_dir: str):
    X_train = np.load(os.path.join(in_dir, "X_train.npy"))
    X_test = np.load(os.path.join(in_dir, "X_test.npy"))
    y_train = np.load(os.path.join(in_dir, "y_train.npy"))
    y_test = np.load(os.path.join(in_dir, "y_test.npy"))
    meta = json.load(open(os.path.join(in_dir, "metadata.json"), encoding="utf-8"))
    return X_train, X_test, y_train, y_test, meta


def train_xgboost(X_train, y_train, task: str):
    from xgboost import XGBClassifier

    params = dict(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,   # L1
        reg_lambda=1.0,  # L2
        random_state=42,
        n_jobs=-1,
        tree_method="hist",
        eval_metric="logloss"
    )
    if task == "binary":
        # scale_pos_weight para lidar com desbalanceamento
        pos = (y_train == 1).sum()
        neg = (y_train == 0).sum()
        if pos > 0:
            params["scale_pos_weight"] = neg / pos

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model, params


def train_lightgbm(X_train, y_train, task: str):
    try:
        from lightgbm import LGBMClassifier
    except Exception as e:
        raise RuntimeError("LightGBM não está instalado. "
                           "Adicione 'lightgbm' ao requirements.txt.") from e

    params = dict(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=64,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1
    )
    if task == "binary":
        params["objective"] = "binary"
    else:
        params["objective"] = "multiclass"
        params["num_class"] = len(np.unique(y_train))

    model = LGBMClassifier(**params)
    model.fit(X_train, y_train)
    return model, params


def kfold_cv(model_ctor, X, y, task: str, n_splits: int = 5) -> dict:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for train_idx, val_idx in skf.split(X, y):
        m, _ = model_ctor(X[train_idx], y[train_idx], task)
        y_pred = m.predict(X[val_idx])
        avg = "binary" if task == "binary" else "macro"
        s = compute_metrics(y[val_idx], y_pred, average=avg)
        scores.append({k: s[k] for k in ["accuracy", "precision", "recall", "f1"]})
    # média por métrica
    agg = {k: float(np.mean([d[k] for d in scores])) for k in scores[0].keys()}
    return {"cv_metrics": agg, "folds": scores}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", default="data/processed")
    ap.add_argument("--out-models", default="models")
    ap.add_argument("--out-reports", default="reports")
    ap.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    args = ap.parse_args()

    ensure_dir(args.out_models)
    ensure_dir(args.out_reports)

    X_train, X_test, y_train, y_test, meta = load_processed(args.in_dir)
    labels = meta["class_names"]
    avg = "binary" if args.task == "binary" else "macro"

    # ----- XGBoost -----
    xgb_cv = kfold_cv(train_xgboost, X_train, y_train, args.task)
    xgb_model, xgb_params = train_xgboost(X_train, y_train, args.task)
    y_pred_xgb = xgb_model.predict(X_test)
    xgb_test = compute_metrics(y_test, y_pred_xgb, average=avg, labels=labels)
    joblib.dump(xgb_model, os.path.join(args.out_models, "xgboost_model.joblib"))
    plot_confusion_matrix(y_test, y_pred_xgb, labels, "XGBoost – Matriz de Confusão",
                          os.path.join(args.out_reports, "cm_xgboost.png"))
    save_json({
        "params": xgb_params,
        "cv": xgb_cv,
        "test": {k: v for k, v in xgb_test.items() if k != "classification_report"},
        "labels": labels
    }, os.path.join(args.out_reports, "metrics_xgboost.json"))
    # salva relatório legível
    with open(os.path.join(args.out_reports, "report_xgboost.txt"), "w", encoding="utf-8") as f:
        f.write(xgb_test["classification_report"])

    # ----- LightGBM -----
    lgb_cv = kfold_cv(train_lightgbm, X_train, y_train, args.task)
    lgb_model, lgb_params = train_lightgbm(X_train, y_train, args.task)
    y_pred_lgb = lgb_model.predict(X_test)
    lgb_test = compute_metrics(y_test, y_pred_lgb, average=avg, labels=labels)
    joblib.dump(lgb_model, os.path.join(args.out_models, "lightgbm_model.joblib"))
    plot_confusion_matrix(y_test, y_pred_lgb, labels, "LightGBM – Matriz de Confusão",
                          os.path.join(args.out_reports, "cm_lightgbm.png"))
    save_json({
        "params": lgb_params,
        "cv": lgb_cv,
        "test": {k: v for k, v in lgb_test.items() if k != "classification_report"},
        "labels": labels
    }, os.path.join(args.out_reports, "metrics_lightgbm.json"))
    with open(os.path.join(args.out_reports, "report_lightgbm.txt"), "w", encoding="utf-8") as f:
        f.write(lgb_test["classification_report"])

    print("[OK] Treinamento finalizado. Modelos em 'models/' e relatórios em 'reports/'.")


if __name__ == "__main__":
    main()
