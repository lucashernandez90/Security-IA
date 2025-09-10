"""
preprocessing.py – Consolida CSVs do CICIDS2017, limpa, normaliza e salva .npy.
Exemplo de uso:

python src/preprocessing.py --raw-dir data/raw/MachineLearningCVE --out-dir data/processed --task binary --scaler minmax

"""
from __future__ import annotations
import argparse
import glob
import json
import os
from typing import List

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils import ensure_dir, save_json


# Colunas que normalmente causam "atalhos" e devem ser removidas
COLUMNS_TO_DROP = [
    "Flow ID", "FlowID",
    "Src IP", "Source IP",
    "Dst IP", "Destination IP",
    "Src Port", "Source Port",
    "Dst Port", "Destination Port",
    "Timestamp", "Timestamp "
]


def read_all_csvs(raw_dir: str) -> pd.DataFrame:
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {raw_dir}")
    dfs = []
    for p in paths:
        df = pd.read_csv(p, low_memory=False)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace("\n", " ").replace("\r", " ") for c in df.columns]
    return df


def map_labels(series: pd.Series, task: str) -> (pd.Series, List[str]):
    """
    task='binary' → 0=Benign, 1=Attack
    task='multiclass' → classes originais
    """
    clean = series.astype(str).str.strip()
    if task == "binary":
        y = (clean != "BENIGN").astype(int)
        labels = ["Benign", "Attack"]
        return y, labels
    # multiclass
    classes = sorted(clean.unique())
    mapping = {cls: i for i, cls in enumerate(classes)}
    y = clean.map(mapping).astype(int)
    return y, classes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-dir", required=True, help="Diretório com CSVs do CICIDS2017")
    ap.add_argument("--out-dir", default="data/processed", help="Saída (.npy, scaler, metadados)")
    ap.add_argument("--task", choices=["binary", "multiclass"], default="binary")
    ap.add_argument("--scaler", choices=["minmax", "standard"], default="minmax")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    # 1) Leitura e normalização de nomes
    df = read_all_csvs(args.raw_dir)
    df = normalize_column_names(df)

    # 2) Identifica coluna de rótulo
    label_col_candidates = [c for c in df.columns if c.lower() == "label"]
    if not label_col_candidates:
        raise RuntimeError("Coluna 'Label' não encontrada nos CSVs.")
    label_col = label_col_candidates[0]

    # 3) Remove colunas indesejadas se existirem
    drop_cols = [c for c in COLUMNS_TO_DROP if c in df.columns]
    df = df.drop(columns=drop_cols, errors="ignore")

    # 4) Converte infinitos para NaN e remove linhas ruins
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()

    # 5) Separa X / y
    y_raw = df[label_col]
    X = df.drop(columns=[label_col])

    # Converte tudo numérico (coerção segura)
    X = X.apply(pd.to_numeric, errors="coerce")
    X = X.dropna()  # remove linhas com não-numéricos
    y_raw = y_raw.loc[X.index]

    # 6) Codifica rótulos
    y, class_names = map_labels(y_raw, args.task)

    # 7) Split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y.values
    )

    # 8) Escalonamento
    scaler = MinMaxScaler() if args.scaler == "minmax" else StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 9) Salva artefatos
    np.save(os.path.join(args.out_dir, "X_train.npy"), X_train_scaled)
    np.save(os.path.join(args.out_dir, "X_test.npy"), X_test_scaled)
    np.save(os.path.join(args.out_dir, "y_train.npy"), y_train)
    np.save(os.path.join(args.out_dir, "y_test.npy"), y_test)

    joblib.dump(scaler, os.path.join(args.out_dir, "scaler.pkl"))

    meta = {
        "task": args.task,
        "scaler": args.scaler,
        "feature_names": X.columns.tolist(),
        "class_names": class_names,
        "dropped_columns": drop_cols,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0])
    }
    save_json(meta, os.path.join(args.out_dir, "metadata.json"))
    print(f"[OK] Processamento concluído. Artefatos em: {args.out_dir}")


if __name__ == "__main__":
    main()
