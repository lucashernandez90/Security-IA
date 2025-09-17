"""
preprocessing.py – Consolida CSVs do CICIDS2017, limpa, normaliza e salva .npy.
Exemplo de uso:

python src/preprocessing.py --raw-dir data/raw/MachineLearningCVE --out-dir data/processed --task multiclass --scaler standard

"""
from __future__ import annotations
import argparse
import glob
import json
import os
from typing import List
import logging

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler

from utils import ensure_dir, save_json

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

COLUMNS_TO_DROP = [
    "Flow ID", "FlowID",
    "Src IP", "Source IP", 
    "Dst IP", "Destination IP",
    "Src Port", "Source Port",
    "Dst Port", "Destination Port",
    "Timestamp", "Timestamp ",
    "Fwd Header Length",  
    "Bwd Header Length",
    "Fwd Header Length.1"
]

def agrupar_classes(label_series: pd.Series) -> pd.Series:
    """Agrupa classes raras em categorias"""
    def agrupar(label):
        label_str = str(label).strip().upper()
        if 'WEB ATTACK' in label_str:
            return 'Web Attack'
        elif 'DOS' in label_str and 'HULK' not in label_str and 'GOLDENEYE' not in label_str:
            return 'DoS Other'
        elif 'PATATOR' in label_str:
            return 'Patator'
        elif label_str in ['HEARTBLEED', 'INFILTRATION']:
            return 'Other Rare Attack'
        elif label_str == 'BENIGN':
            return 'BENIGN'
        elif label_str in ['BOT', 'DDOS', 'DOS HULK', 'DOS GOLDENEYE']:
            return label_str
        else:
            return 'Other Rare Attack'
    
    return label_series.apply(agrupar)

def read_all_csvs(raw_dir: str) -> pd.DataFrame:
    """Lê e concatena todos os CSVs do diretório"""
    paths = sorted(glob.glob(os.path.join(raw_dir, "*.csv")))
    if not paths:
        raise FileNotFoundError(f"Nenhum CSV encontrado em {raw_dir}")
    
    dfs = []
    for i, p in enumerate(paths):
        try:
            logger.info(f"Lendo arquivo {i+1}/{len(paths)}: {os.path.basename(p)}")
            df = pd.read_csv(p, low_memory=False, encoding='utf-8')
            logger.info(f"Arquivo {os.path.basename(p)}: {df.shape[0]} linhas, {df.shape[1]} colunas")
            dfs.append(df)
        except Exception as e:
            logger.warning(f"Erro ao ler {p}: {e}")
            continue
    
    if not dfs:
        raise RuntimeError("Nenhum CSV foi lido com sucesso")
    
    result = pd.concat(dfs, ignore_index=True)
    logger.info(f"Total de dados concatenados: {result.shape[0]} linha, {result.shape[1]} colunas")
    return result

def validate_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Valida e limpa o dataframe"""
    original_shape = df.shape
    logger.info(f"Shape original: {original_shape}")
    
    # Remove linhas completamente vazias
    df = df.dropna(how='all')
    logger.info(f"Após remover linhas vazias: {df.shape}")
    
    # Remove colunas duplicadas
    df = df.loc[:, ~df.columns.duplicated()]
    logger.info(f"Após remover colunas duplicadas: {df.shape}")
    
    # Remove colunas com apenas um valor único
    single_value_cols = []
    for col in df.columns:
        if df[col].nunique() <= 1:
            single_value_cols.append(col)
    
    if single_value_cols:
        logger.warning(f"Removendo colunas com único valor: {single_value_cols}")
        df = df.drop(columns=single_value_cols)
    
    logger.info(f"Shape final validado: {df.shape}")
    return df

def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nomes das colunas"""
    df = df.copy()
    original_cols = df.columns.tolist()
    
    df.columns = [c.strip().replace("\n", " ").replace("\r", " ").replace("  ", " ") for c in df.columns]
    
    # Log de mudanças nos nomes
    for orig, new in zip(original_cols, df.columns):
        if orig != new:
            logger.debug(f"Coluna renomeada: '{orig}' -> '{new}'")
    
    return df

def map_labels(series: pd.Series, task: str) -> (pd.Series, List[str]):
    """
    task='binary' → 0=Benign, 1=Attack  
    task='multiclass' → classes originais
    """
    clean = series.astype(str).str.strip().str.upper()
    
    if task == "binary":
        y = (clean != "BENIGN").astype(int)
        labels = ["Benign", "Attack"]
        
        benign_count = (y == 0).sum()
        attack_count = (y == 1).sum()
        logger.info(f"Distribuição binária - Benign: {benign_count}, Attack: {attack_count}")
        
        return y, labels
    
    # multiclass
    classes = sorted(clean.unique())
    mapping = {cls: i for i, cls in enumerate(classes)}
    y = clean.map(mapping).astype(int)
    
    class_counts = y.value_counts().to_dict()
    logger.info(f"Distribuição multiclasse: {class_counts}")
    logger.info(f"Classes: {classes}")
    
    return y, classes

def analyze_missing_data(df: pd.DataFrame) -> dict:
    """Analisa dados faltantes no dataframe"""
    total_cells = df.size
    missing_cells = df.isnull().sum().sum()
    missing_percentage = (missing_cells / total_cells) * 100
    
    missing_by_col = df.isnull().sum()
    high_missing_cols = missing_by_col[missing_by_col > 0]
    
    analysis = {
        "total_cells": total_cells,
        "missing_cells": int(missing_cells),
        "missing_percentage": float(missing_percentage),
        "columns_with_missing": int((missing_by_col > 0).sum()),
        "high_missing_columns": high_missing_cols.to_dict()
    }
    
    logger.info(f"Células com missing values: {missing_cells}/{total_cells} ({missing_percentage:.2f}%)")
    if not high_missing_cols.empty:
        logger.warning(f"Colunas com missing values: {high_missing_cols.to_dict()}")
    
    return analysis

def main():
    ap = argparse.ArgumentParser(description="Pré-processamento de dados do CICIDS2017")
    ap.add_argument("--raw-dir", required=True, help="Diretório com CSVs do CICIDS2017")
    ap.add_argument("--out-dir", default="data/processed", help="Saída (.npy, scaler, metadados)")
    ap.add_argument("--task", choices=["binary", "multiclass"], default="multiclass", help="Tipo de tarefa")
    ap.add_argument("--scaler", choices=["minmax", "standard"], default="standard", help="Tipo de scaler")
    ap.add_argument("--test-size", type=float, default=0.2, help="Tamanho do conjunto de teste")
    ap.add_argument("--random-state", type=int, default=42, help="Seed para reproducibilidade")
    ap.add_argument("--min-samples", type=int, default=100, help="Mínimo de amostras por classe")
    args = ap.parse_args()

    ensure_dir(args.out_dir)
    logger.info(f"Iniciando pré-processamento com parâmetros: {vars(args)}")

    try:
        # 1) Leitura e normalização de nomes
        logger.info("Lendo e concatenando CSVs...")
        df = read_all_csvs(args.raw_dir)
        df = normalize_column_names(df)
        df = validate_dataframe(df)

        # 2) Análise de dados faltantes
        missing_analysis = analyze_missing_data(df)
        
        # 3) Identifica coluna de rótulo
        label_col_candidates = [c for c in df.columns if c.lower() == "label"]
        if not label_col_candidates:
            raise RuntimeError("Coluna 'Label' não encontrada nos CSVs.")
        label_col = label_col_candidates[0]
        logger.info(f"Coluna de label identificada: {label_col}")

        # 4) Remove colunas indesejadas se existirem
        drop_cols = [c for c in COLUMNS_TO_DROP if c in df.columns]
        if drop_cols:
            logger.info(f"Removendo colunas: {drop_cols}")
            df = df.drop(columns=drop_cols, errors="ignore")
        else:
            logger.info("Nenhuma coluna para remover encontrada")

        # 5) Converte infinitos para NaN e remove linhas ruins
        logger.info("Lidando com valores infinitos e NaN...")
        df = df.replace([np.inf, -np.inf], np.nan)
        original_size = len(df)
        df = df.dropna()
        removed_count = original_size - len(df)
        logger.info(f"Linhas removidas devido a NaN/inf: {removed_count}")

        # 6) Separa X / y
        y_raw = df[label_col]
        X = df.drop(columns=[label_col])

        # 6.5) Aplicar agrupamento de classes para multiclass
        if args.task == "multiclass":
            logger.info("Agrupando classes raras...")
            y_raw = agrupar_classes(y_raw)
            logger.info(f"Distribuição após agrupamento: {y_raw.value_counts().to_dict()}")

        # 7) Converte tudo numérico (coerção segura)
        logger.info("Convertendo features para numérico...")
        X = X.apply(pd.to_numeric, errors="coerce")
        
        # Remove linhas com não-numéricos
        non_numeric_mask = X.isnull().any(axis=1)
        if non_numeric_mask.any():
            logger.warning(f"Removendo {non_numeric_mask.sum()} linhas com valores não numéricos")
            X = X[~non_numeric_mask]
            y_raw = y_raw[~non_numeric_mask]

        # 8) Codifica rótulos
        logger.info("Codificando labels...")
        y, class_names = map_labels(y_raw, args.task)

        # 9) Split estratificado
        logger.info("Dividindo em treino e teste...")
        X_train, X_test, y_train, y_test = train_test_split(
            X.values, y.values,
            test_size=args.test_size,
            random_state=args.random_state,
            stratify=y.values
        )

        logger.info(f"Split final - Treino: {X_train.shape}, Teste: {X_test.shape}")

        # 10) Escalonamento
        logger.info(f"Aplicando scaler: {args.scaler}")
        scaler = MinMaxScaler() if args.scaler == "minmax" else StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 11) Salva artefatos
        logger.info("Salvando artefatos...")
        np.save(os.path.join(args.out_dir, "X_train_balanced.npy"), X_train_scaled)
        np.save(os.path.join(args.out_dir, "X_test_scaled.npy"), X_test_scaled)
        np.save(os.path.join(args.out_dir, "y_train_balanced.npy"), y_train)
        np.save(os.path.join(args.out_dir, "y_test.npy"), y_test)

        joblib.dump(scaler, os.path.join(args.out_dir, "scaler.pkl"))

        meta = {
            "task": args.task,
            "scaler": args.scaler,
            "feature_names": X.columns.tolist(),
            "class_names": class_names,
            "dropped_columns": drop_cols,
            "n_train": int(X_train.shape[0]),
            "n_test": int(X_test.shape[0]),
            "missing_data_analysis": missing_analysis,
            "label_distribution": {
                "train": {str(cls): int(count) for cls, count in zip(*np.unique(y_train, return_counts=True))},
                "test": {str(cls): int(count) for cls, count in zip(*np.unique(y_test, return_counts=True))}
            },
            "preprocessing_parameters": vars(args)
        }
        
        save_json(meta, os.path.join(args.out_dir, "metadata.json"))
        
        logger.info(f"[OK] Processamento concluído. Artefatos em: {args.out_dir}")
        logger.info(f"Distribuição final - Treino: {np.bincount(y_train)}, Teste: {np.bincount(y_test)}")

    except Exception as e:
        logger.error(f"Erro no pré-processamento: {e}")
        raise

if __name__ == "__main__":
    main()