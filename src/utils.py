from __future__ import annotations
import json
import os
import logging
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics
import seaborn as sns

# Configuração de logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def ensure_dir(path: str) -> None:
    """Cria diretório se não existir."""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.error(f"Erro ao criar diretório {path}: {e}")
        raise

def save_json(obj: dict, path: str, indent: int = 2) -> None:
    """
    Salva objeto JSON com tratamento de erros.
    
    Args:
        obj: Dicionário a ser salvo
        path: Caminho do arquivo
        indent: Indentação do JSON
    """
    try:
        ensure_dir(os.path.dirname(path) or ".")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=indent, ensure_ascii=False, default=str)
        logger.debug(f"JSON salvo com sucesso: {path}")
    except TypeError as e:
        logger.error(f"Erro de serialização JSON: {e}")
        raise
    except Exception as e:
        logger.error(f"Erro ao salvar JSON em {path}: {e}")
        raise

def load_json(path: str) -> dict:
    """
    Carrega arquivo JSON com tratamento de erros.
    
    Args:
        path: Caminho do arquivo JSON
        
    Returns:
        Dicionário com os dados carregados
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Arquivo JSON não encontrado: {path}")
        raise
    except json.JSONDecodeError as e:
        logger.error(f"Erro ao decodificar JSON {path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Erro ao carregar JSON {path}: {e}")
        raise

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    average: str = "binary",
                    labels: Optional[List[str]] = None,
                    y_prob: Optional[np.ndarray] = None) -> Dict:
    """
    Retorna métricas padrão (accuracy, precision, recall, f1) + relatório.
    
    Args:
        y_true: Valores verdadeiros
        y_pred: Valores preditos
        average: Tipo de average para métricas multiclasse
        labels: Nomes das classes
        y_prob: Probabilidades das predições (opcional)
        
    Returns:
        Dicionário com métricas e relatório
    """
    try:
        # Validação de inputs
        if len(y_true) != len(y_pred):
            raise ValueError(f"Tamanhos diferentes: y_true({len(y_true)}), y_pred({len(y_pred)})")
        
        if len(y_true) == 0:
            raise ValueError("Arrays vazios")
        
        # Métricas básicas
        acc = metrics.accuracy_score(y_true, y_pred)
        
        if average == "binary":
            prec = metrics.precision_score(y_true, y_pred, zero_division=0)
            rec = metrics.recall_score(y_true, y_pred, zero_division=0)
            f1 = metrics.f1_score(y_true, y_pred, zero_division=0)
        else:
            prec = metrics.precision_score(y_true, y_pred, average=average, zero_division=0)
            rec = metrics.recall_score(y_true, y_pred, average=average, zero_division=0)
            f1 = metrics.f1_score(y_true, y_pred, average=average, zero_division=0)
        
        report = metrics.classification_report(y_true, y_pred, target_names=labels, zero_division=0)
        
        #Matriz de confusão em formato numérico
        cm = metrics.confusion_matrix(y_true, y_pred)
        
        # Métricas adicionais
        result = {
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1": float(f1),
            "classification_report": report,
            "confusion_matrix": cm.tolist(),
            "n_samples": int(len(y_true)),
            "n_classes": int(len(np.unique(y_true))),
            "class_distribution": {
                str(cls): int(count) for cls, count in zip(*np.unique(y_true, return_counts=True))
            }
        }
        
        # Adicionar métricas baseadas em probabilidade se disponível
        if y_prob is not None:
            try:
                if average == "binary":
                    roc_auc = metrics.roc_auc_score(y_true, y_prob[:, 1] if y_prob.shape[1] > 1 else y_prob)
                else:
                    roc_auc = metrics.roc_auc_score(y_true, y_prob, multi_class='ovr')
                
                result.update({
                    "roc_auc": float(roc_auc),
                    "log_loss": float(metrics.log_loss(y_true, y_prob)),
                    "average_confidence": float(np.mean(np.max(y_prob, axis=1)))
                })
            except Exception as e:
                logger.warning(f"Erro ao calcular métricas de probabilidade: {e}")
                result["probability_metrics_error"] = str(e)
        
        result["balanced_accuracy"] = float(metrics.balanced_accuracy_score(y_true, y_pred))
        
        return result
        
    except Exception as e:
        logger.error(f"Erro no cálculo de métricas: {e}")
        raise

def plot_confusion_matrix(y_true: np.ndarray,
                          y_pred: np.ndarray,
                          labels: List[str],
                          title: str,
                          out_path: str,
                          normalize: bool = True,
                          figsize: Tuple[int, int] = (10, 8),
                          dpi: int = 300) -> None:
    """
    Gera e salva a matriz de confusão com melhor visualização.
    
    Args:
        y_true: Valores verdadeiros
        y_pred: Valores preditos
        labels: Nomes das classes
        title: Título do gráfico
        out_path: Caminho de saída
        normalize: Se True, normaliza por linha
        figsize: Tamanho da figura
        dpi: Resolução da imagem
    """
    try:
        # Calcular matriz de confusão
        cm = metrics.confusion_matrix(y_true, y_pred, labels=list(range(len(labels))))
        
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
            cm = np.nan_to_num(cm)  # Handle division by zero
            fmt = '.2f'
        else:
            fmt = 'd'
        
        fig, ax = plt.subplots(figsize=figsize)
        
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Proporção' if normalize else 'Contagem'},
                   ax=ax)

        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.set_xlabel('Predito', fontsize=12, fontweight='bold')
        ax.set_ylabel('Verdadeiro', fontsize=12, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        plt.tight_layout()
        
        # Garantir que o diretório existe
        ensure_dir(os.path.dirname(out_path) or ".")
        
        # Salvar figura
        plt.savefig(out_path, dpi=dpi, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        plt.close(fig)
        
        logger.debug(f"Matriz de confusão salva: {out_path}")
        
    except Exception as e:
        logger.error(f"Erro ao plotar matriz de confusão: {e}")
        raise

def plot_roc_curve(y_true: np.ndarray, 
                   y_prob: np.ndarray, 
                   labels: List[str],
                   title: str,
                   out_path: str,
                   figsize: Tuple[int, int] = (10, 8)) -> None:
    """
    Plota curva ROC para problemas multiclasse.
    
    Args:
        y_true: Valores verdadeiros
        y_prob: Probabilidades das predições
        labels: Nomes das classes
        title: Título do gráfico
        out_path: Caminho de saída
        figsize: Tamanho da figura
    """
    try:
        n_classes = len(labels)
        
        #Calcula ROC
        fpr = dict()
        tpr = dict()
        roc_auc = dict()
        
        for i in range(n_classes):
            fpr[i], tpr[i], _ = metrics.roc_curve(y_true == i, y_prob[:, i])
            roc_auc[i] = metrics.auc(fpr[i], tpr[i])
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = plt.cm.Set3(np.linspace(0, 1, n_classes))
        for i, color in zip(range(n_classes), colors):
            ax.plot(fpr[i], tpr[i], color=color, lw=2,
                   label=f'{labels[i]} (AUC = {roc_auc[i]:.2f})')
        
        ax.plot([0, 1], [0, 1], 'k--', lw=2)
        
        # Configurações do gráfico
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('Taxa de Falsos Positivos', fontsize=12)
        ax.set_ylabel('Taxa de Verdadeiros Positivos', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold')
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        # Salvar
        ensure_dir(os.path.dirname(out_path) or ".")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        logger.debug(f"Curva ROC salva: {out_path}")
        
    except Exception as e:
        logger.error(f"Erro ao plotar curva ROC: {e}")
        raise

def save_feature_importance(feature_importances: np.ndarray,
                           feature_names: List[str],
                           out_path: str,
                           top_n: int = 20) -> None:
    """
    Salva importância das features em JSON e opcionalmente plota gráfico.
    
    Args:
        feature_importances: Array com importância das features
        feature_names: Nomes das features
        out_path: Caminho de saída (sem extensão)
        top_n: Número de top features para incluir no JSON
    """
    try:
        #Criar dicionário com importância das features
        importance_dict = {
            name: float(imp) 
            for name, imp in zip(feature_names, feature_importances)
        }
        
        # Ordenar por importância
        sorted_importance = dict(sorted(
            importance_dict.items(), 
            key=lambda x: x[1], 
            reverse=True
        ))
        
        #Salva JSON completo
        save_json(sorted_importance, f"{out_path}.json")
        
        #Salva top N features
        top_features = dict(list(sorted_importance.items())[:top_n])
        save_json(top_features, f"{out_path}_top{top_n}.json")
        
        logger.debug(f"Importância das features salva: {out_path}.json")
        
    except Exception as e:
        logger.error(f"Erro ao salvar importância das features: {e}")
        raise

def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:

    """ Calcula pesos de classes para datasets desbalanceados."""
    try:
        class_counts = np.bincount(y)
        n_classes = len(class_counts)
        total_samples = len(y)
        
        weights = {}
        for i in range(n_classes):
            if class_counts[i] > 0:
                weights[i] = total_samples / (n_classes * class_counts[i])
            else:
                weights[i] = 0.0
                
        return weights
        
    except Exception as e:
        logger.error(f"Erro ao calcular pesos de classes: {e}")
        raise

def time_execution(func):
    """ Decorator para medir tempo de execução de funções. """
    import time
    from functools import wraps
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        
        logger.info(f"Função {func.__name__} executada em {end_time - start_time:.2f} segundos")
        return result
        
    return wrapper

# Exemplo de uso do decorator (opcional)
@time_execution
def example_slow_function():
    """Função de exemplo para demonstrar o decorator de tempo."""
    import time
    time.sleep(2)
    return "Done"