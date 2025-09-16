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
import logging
from sklearn.model_selection import StratifiedKFold
from utils import ensure_dir, save_json, compute_metrics, plot_confusion_matrix

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_processed(in_dir: str):
    """Carrega dados pré-processados + metadados"""
    try:
        X_train = np.load(os.path.join(in_dir, "X_train.npy"))
        X_test = np.load(os.path.join(in_dir, "X_test.npy"))
        y_train = np.load(os.path.join(in_dir, "y_train.npy"))
        y_test = np.load(os.path.join(in_dir, "y_test.npy"))
        meta = json.load(open(os.path.join(in_dir, "metadata.json"), encoding="utf-8"))
        logger.info(f"Dados carregados: X_train={X_train.shape}, X_test={X_test.shape}")
        return X_train, X_test, y_train, y_test, meta
    except Exception as e:
        logger.error(f"Erro ao carregar dados processados: {e}")
        raise

def train_xgboost(X_train, y_train, task: str):
    """Treina modelo XGBoost"""
    try:
        from xgboost import XGBClassifier
        params = dict(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            tree_method="hist",
            eval_metric="logloss",
            verbosity=0
        )
        if task == "binary":
            pos, neg = (y_train == 1).sum(), (y_train == 0).sum()
            if pos > 0:
                params["scale_pos_weight"] = neg / pos
                logger.info(f"Scale pos weight calculado: {params['scale_pos_weight']:.2f}")

        model = XGBClassifier(**params)
        model.fit(X_train, y_train)
        logger.info("XGBoost treinado com sucesso")
        return model, params
    except Exception as e:
        logger.error(f"Erro no treinamento XGBoost: {e}")
        raise

def train_lightgbm(X_train, y_train, task: str):
    """Treina modelo LightGBM"""
    try:
        from lightgbm import LGBMClassifier
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
            n_jobs=-1,
            verbose=-1
        )
        if task == "binary":
            params["objective"] = "binary"
            params["is_unbalance"] = True
        else:
            params["objective"] = "multiclass"
            params["num_class"] = len(np.unique(y_train))

        model = LGBMClassifier(**params)
        model.fit(X_train, y_train)
        logger.info("LightGBM treinado com sucesso")
        return model, params
    except Exception as e:
        logger.error(f"Erro no treinamento LightGBM: {e}")
        raise

def kfold_cv(model_ctor, X, y, task: str, n_splits: int = 5):
    """Executa validação cruzada estratificada e retorna métricas médias"""
    try:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        scores = []
        
        logger.info(f"Iniciando {n_splits}-fold cross validation...")
        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            logger.info(f"Processando fold {fold + 1}/{n_splits}")
            m, _ = model_ctor(X[train_idx], y[train_idx], task)
            y_pred = m.predict(X[val_idx])
            avg = "binary" if task == "binary" else "macro"
            s = compute_metrics(y[val_idx], y_pred, average=avg)
            scores.append({k: s[k] for k in ["accuracy", "precision", "recall", "f1"]})
            
            logger.info(f"Fold {fold + 1}: Accuracy={s['accuracy']:.4f}, F1={s['f1']:.4f}")
        
        agg = {k: float(np.mean([d[k] for d in scores])) for k in scores[0].keys()}
        logger.info(f"CV final - Accuracy: {agg['accuracy']:.4f}, F1: {agg['f1']:.4f}")
        return {"cv_metrics": agg, "folds": scores}
    except Exception as e:
        logger.error(f"Erro na validação cruzada: {e}")
        raise

def save_model_artifacts(name, model, params, cv, X_test, y_test, labels, avg, out_models, out_reports):
    """Salva modelo, métricas e relatórios"""
    try:
        from sklearn.metrics import classification_report

        # Predições
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
        test_metrics = compute_metrics(y_test, y_pred, average=avg, labels=labels)

        # Modelo
        model_path = os.path.join(out_models, f"{name}.joblib")
        joblib.dump(model, model_path)
        logger.info(f"Modelo salvo: {model_path}")

        # Matriz de confusão
        cm_path = os.path.join(out_reports, f"cm_{name}.png")
        plot_confusion_matrix(y_test, y_pred, labels,
                              f"{name} – Matriz de Confusão",
                              cm_path)
        logger.info(f"Matriz de confusão salva: {cm_path}")

        # JSON com hiperparâmetros e métricas
        metrics_data = {
            "params": params,
            "cv": cv,
            "test": {k: v for k, v in test_metrics.items() if k != "classification_report"},
            "test_probabilities": {
                "mean_confidence": float(np.mean(np.max(y_prob, axis=1))) if y_prob is not None else None
            },
            "labels": labels,
            "model_info": {
                "n_features": X_test.shape[1],
                "n_test_samples": len(y_test),
                "feature_importances": model.feature_importances_.tolist() if hasattr(model, 'feature_importances_') else None
            }
        }
        
        metrics_path = os.path.join(out_reports, f"metrics_{name}.json")
        save_json(metrics_data, metrics_path)
        logger.info(f"Métricas salvas: {metrics_path}")

        # Relatório detalhado
        report_path = os.path.join(out_reports, f"report_{name}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(test_metrics["classification_report"])
        logger.info(f"Relatório salvo: {report_path}")

        return test_metrics
        
    except Exception as e:
        logger.error(f"Erro ao salvar artefatos do modelo {name}: {e}")
        raise

def main():
    ap = argparse.ArgumentParser(description="Treina modelos de detecção de intrusão")
    ap.add_argument("--in-dir", default="data/processed", help="Diretório com dados processados")
    ap.add_argument("--out-models", default="models", help="Diretório de saída para modelos")
    ap.add_argument("--out-reports", default="reports", help="Diretório de saída para relatórios")
    ap.add_argument("--task", choices=["binary", "multiclass"], default="binary", help="Tipo de tarefa de classificação")
    ap.add_argument("--cv-folds", type=int, default=5, help="Número de folds para cross-validation")
    args = ap.parse_args()

    try:
        ensure_dir(args.out_models)
        ensure_dir(args.out_reports)

        logger.info("Iniciando processo de treinamento...")
        logger.info(f"Parâmetros: task={args.task}, cv_folds={args.cv_folds}")

        X_train, X_test, y_train, y_test, meta = load_processed(args.in_dir)
        labels = meta["class_names"]
        avg = "binary" if args.task == "binary" else "macro"

        logger.info(f"Distribuição das classes - Treino: {np.bincount(y_train)}, Teste: {np.bincount(y_test)}")

        # XGBoost
        logger.info("=" * 50)
        logger.info("TREINANDO XGBOOST")
        logger.info("=" * 50)
        
        xgb_cv = kfold_cv(train_xgboost, X_train, y_train, args.task, args.cv_folds)
        xgb_model, xgb_params = train_xgboost(X_train, y_train, args.task)
        xgb_test_metrics = save_model_artifacts("xgboost", xgb_model, xgb_params, xgb_cv,
                         X_test, y_test, labels, avg,
                         args.out_models, args.out_reports)

        # LightGBM
        logger.info("=" * 50)
        logger.info("TREINANDO LIGHTGBM")
        logger.info("=" * 50)
        
        lgb_cv = kfold_cv(train_lightgbm, X_train, y_train, args.task, args.cv_folds)
        lgb_model, lgb_params = train_lightgbm(X_train, y_train, args.task)
        lgb_test_metrics = save_model_artifacts("lightgbm", lgb_model, lgb_params, lgb_cv,
                         X_test, y_test, labels, avg,
                         args.out_models, args.out_reports)

        # Comparação final
        logger.info("=" * 50)
        logger.info("COMPARAÇÃO FINAL DOS MODELOS")
        logger.info("=" * 50)
        logger.info(f"XGBoost - Accuracy: {xgb_test_metrics['accuracy']:.4f}, F1: {xgb_test_metrics['f1']:.4f}")
        logger.info(f"LightGBM - Accuracy: {lgb_test_metrics['accuracy']:.4f}, F1: {lgb_test_metrics['f1']:.4f}")

        # Salvar comparação
        comparison = {
            "xgboost": {
                "accuracy": xgb_test_metrics["accuracy"],
                "f1": xgb_test_metrics["f1"],
                "precision": xgb_test_metrics["precision"],
                "recall": xgb_test_metrics["recall"]
            },
            "lightgbm": {
                "accuracy": lgb_test_metrics["accuracy"],
                "f1": lgb_test_metrics["f1"],
                "precision": lgb_test_metrics["precision"],
                "recall": lgb_test_metrics["recall"]
            },
            "best_model": "xgboost" if xgb_test_metrics["f1"] > lgb_test_metrics["f1"] else "lightgbm"
        }
        
        save_json(comparison, os.path.join(args.out_reports, "model_comparison.json"))
        
        logger.info("[OK] Treinamento finalizado. Modelos em 'models/' e relatórios em 'reports/'.")

    except Exception as e:
        logger.error(f"Erro no processo de treinamento: {e}")
        raise

if __name__ == "__main__":
    main()