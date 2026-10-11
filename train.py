"""
train.py — Entrenamiento parametrizado del modelo de churn (Telco Customer Churn)
instrumentado con MLflow Tracking.

Uso básico (valores por defecto):
    python train.py

Ejemplo cambiando hiperparámetros:
    python train.py --C 0.1 --penalty l1 --class-weight balanced --seed 42

Cada ejecución crea un run dentro del experimento indicado en --experiment-name
y registra parámetros, métricas, artefactos y el modelo entrenado.
"""

import argparse
import hashlib
import json
import os
import random
import tempfile
import urllib.request
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin ventana: el script corre igual en terminal o servidor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import sklearn

SKLEARN_VERSION = tuple(int(v) for v in sklearn.__version__.split(".")[:2])

DATA_URL = (
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/"
    "master/data/Telco-Customer-Churn.csv"
)
DEFAULT_DATA_PATH = Path("data") / "Telco-Customer-Churn.csv"
TARGET = "Churn"
NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]


# --------------------------------------------------------------------------- #
# 1. Argumentos de línea de comandos
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    """Define los hiperparámetros y opciones configurables desde la terminal."""
    p = argparse.ArgumentParser(
        description="Entrena una regresión logística para predecir churn y registra el run en MLflow.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # Hiperparámetros del modelo
    p.add_argument("--C", type=float, default=1.0,
                   help="Inverso de la fuerza de regularización (menor = más regularización).")
    p.add_argument("--penalty", choices=["l1", "l2"], default="l2",
                   help="Tipo de regularización.")
    p.add_argument("--class-weight", choices=["none", "balanced"], default="none",
                   help="'balanced' pondera las clases inversamente a su frecuencia.")
    p.add_argument("--max-iter", type=int, default=1000,
                   help="Iteraciones máximas del optimizador.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Umbral de probabilidad para clasificar como churn.")
    # Datos y reproducibilidad
    p.add_argument("--test-size", type=float, default=0.2, help="Proporción del conjunto de prueba.")
    p.add_argument("--seed", type=int, default=42, help="Semilla aleatoria global.")
    p.add_argument("--data-path", type=Path, default=DEFAULT_DATA_PATH,
                   help="Ruta local del CSV (se descarga si no existe).")
    # MLflow
    p.add_argument("--experiment-name", default="TC5061_Telco_Churn_LogReg",
                   help="Experimento de MLflow donde se agrupan los runs.")
    p.add_argument("--run-name", default=None, help="Nombre del run (opcional).")
    p.add_argument("--tracking-uri", default=os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000"),
                   help="URI del servidor de MLflow.")
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# 2. Reproducibilidad
# --------------------------------------------------------------------------- #
def set_seed(seed):
    """Fija las semillas de Python y NumPy; la misma semilla se pasa a la partición y al modelo."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)


# --------------------------------------------------------------------------- #
# 3. Datos: carga y limpieza (decisiones documentadas en el notebook de EDA)
# --------------------------------------------------------------------------- #
def load_data(path):
    """Descarga el CSV si no existe localmente y lo carga en un DataFrame."""
    path = Path(path)
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Descargando dataset en {path} ...")
        urllib.request.urlretrieve(DATA_URL, path)
    return pd.read_csv(path)


def file_md5(path):
    """Hash MD5 del CSV: permite detectar si los datos cambiaron entre corridas."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


def clean_data(df):
    """Aplica la limpieza definida en el EDA y devuelve X, y."""
    df = df.copy()
    # TotalCharges viene como texto: 11 valores en blanco que corresponden a clientes con tenure = 0
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"].astype(str).str.strip().replace("", np.nan))
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    # SeniorCitizen es categórica aunque venga codificada como 0/1
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    # Sin duplicados ni outliers por IQR (ver notebook); se elimina el identificador
    df = df.drop(columns="customerID")
    y = (df[TARGET] == "Yes").astype(int)
    X = df.drop(columns=TARGET)
    return X, y


def split_data(X, y, test_size, seed):
    """Partición estratificada train/test."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=seed)


# --------------------------------------------------------------------------- #
# 4. Modelo
# --------------------------------------------------------------------------- #
def build_pipeline(X, args):
    """Pipeline de preprocesamiento + regresión logística con los hiperparámetros recibidos."""
    cat_cols = [c for c in X.columns if c not in NUM_COLS]
    preprocessor = ColumnTransformer([
        ("num", StandardScaler(), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), cat_cols),
    ])
    # scikit-learn >= 1.8 expresa el tipo de regularización con l1_ratio (1 = L1, 0 = L2)
    if SKLEARN_VERSION >= (1, 8):
        reg = {"l1_ratio": 1.0 if args.penalty == "l1" else 0.0}
    else:
        reg = {"penalty": args.penalty}
    clf = LogisticRegression(
        C=args.C,
        solver="liblinear",  # soporta l1 y l2: mismo solver en todas las corridas para compararlas
        class_weight=None if args.class_weight == "none" else "balanced",
        max_iter=args.max_iter,
        random_state=args.seed,
        **reg,
    )
    return Pipeline([("preprocesamiento", preprocessor), ("clasificador", clf)])


def train(pipeline, X_train, y_train, seed):
    """Calcula ROC-AUC con validación cruzada (5 folds) y entrena con todo el set de entrenamiento."""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    cv_auc = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="roc_auc")
    pipeline.fit(X_train, y_train)
    return pipeline, cv_auc


# --------------------------------------------------------------------------- #
# 5. Evaluación y artefactos
# --------------------------------------------------------------------------- #
def evaluate(pipeline, X_test, y_test, threshold):
    """Métricas de clasificación sobre prueba usando el umbral indicado."""
    proba = pipeline.predict_proba(X_test)[:, 1]
    pred = (proba >= threshold).astype(int)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred),
        "f1": f1_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, proba),
        "pr_auc": average_precision_score(y_test, proba),
    }
    return metrics, pred, proba


def save_artifacts(out_dir, y_test, pred, proba, metrics, params):
    """Genera matriz de confusión, curva ROC, reporte y archivo de métricas en out_dir."""
    out_dir = Path(out_dir)
    paths = {}

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        y_test, pred, display_labels=["No churn", "Churn"], cmap="Blues", colorbar=False, ax=ax)
    ax.set_title("Matriz de confusión (test)")
    fig.tight_layout()
    paths["confusion"] = out_dir / "confusion_matrix.png"
    fig.savefig(paths["confusion"], dpi=120)
    plt.close(fig)

    fpr, tpr, _ = roc_curve(y_test, proba)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.plot(fpr, tpr, color="#2a78d6", lw=2, label=f"AUC = {metrics['roc_auc']:.3f}")
    ax.plot([0, 1], [0, 1], ls="--", color="#8a8984", lw=1, label="Azar")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos")
    ax.set_title("Curva ROC (test)")
    ax.legend(loc="lower right")
    fig.tight_layout()
    paths["roc"] = out_dir / "roc_curve.png"
    fig.savefig(paths["roc"], dpi=120)
    plt.close(fig)

    paths["report"] = out_dir / "classification_report.txt"
    paths["report"].write_text(
        classification_report(y_test, pred, target_names=["No churn", "Churn"], digits=3),
        encoding="utf-8")

    tn, fp, fn, tp = confusion_matrix(y_test, pred).ravel()
    paths["metrics"] = out_dir / "metrics.json"
    paths["metrics"].write_text(json.dumps({
        "params": params,
        "metrics": {k: round(float(v), 6) for k, v in metrics.items()},
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }, indent=2, ensure_ascii=False), encoding="utf-8")
    return paths


# --------------------------------------------------------------------------- #
# 6. MLflow
# --------------------------------------------------------------------------- #
def configure_mlflow(tracking_uri, experiment_name):
    """Conecta con el servidor de MLflow; si no responde, usa un archivo SQLite local."""
    if tracking_uri.startswith("http"):
        try:
            urllib.request.urlopen(f"{tracking_uri.rstrip('/')}/health", timeout=3)
        except Exception:
            print(f"[aviso] No hay servidor en {tracking_uri}; se usará sqlite:///mlflow.db")
            tracking_uri = "sqlite:///mlflow.db"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(experiment_name)
    return tracking_uri


def default_run_name(args):
    cw = "bal" if args.class_weight == "balanced" else "nobal"
    return f"logreg_{args.penalty}_C{args.C:g}_{cw}_thr{args.threshold:g}"


def main(argv=None):
    args = parse_args(argv)
    set_seed(args.seed)

    X, y = clean_data(load_data(args.data_path))
    X_train, X_test, y_train, y_test = split_data(X, y, args.test_size, args.seed)

    uri = configure_mlflow(args.tracking_uri, args.experiment_name)
    run_name = args.run_name or default_run_name(args)

    params = {
        "model": "LogisticRegression",
        "C": args.C,
        "penalty": args.penalty,
        "class_weight": args.class_weight,
        "max_iter": args.max_iter,
        "threshold": args.threshold,
        "solver": "liblinear",
        "test_size": args.test_size,
        "seed": args.seed,
        "n_train": len(X_train),
        "n_test": len(X_test),
        "data_md5": file_md5(args.data_path),
    }

    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({"curso": "TC5061", "dataset": "Telco Customer Churn (IBM)",
                         "tipo_problema": "clasificacion_binaria"})

        # Hiperparámetros y configuración
        for key, value in params.items():
            mlflow.log_param(key, value)

        # Entrenamiento y evaluación
        pipeline = build_pipeline(X_train, args)
        pipeline, cv_auc = train(pipeline, X_train, y_train, args.seed)
        metrics, pred, proba = evaluate(pipeline, X_test, y_test, args.threshold)

        # Métricas
        for key, value in metrics.items():
            mlflow.log_metric(f"test_{key}", value)
        mlflow.log_metric("cv_roc_auc_mean", cv_auc.mean())
        mlflow.log_metric("cv_roc_auc_std", cv_auc.std())

        # Artefactos
        with tempfile.TemporaryDirectory() as tmp:
            paths = save_artifacts(tmp, y_test, pred, proba, metrics, params)
            mlflow.log_artifact(str(paths["confusion"]), artifact_path="graficas")
            mlflow.log_artifact(str(paths["roc"]), artifact_path="graficas")
            mlflow.log_artifact(str(paths["report"]), artifact_path="reportes")
            mlflow.log_artifact(str(paths["metrics"]), artifact_path="reportes")

        # Modelo entrenado (pipeline completo) con firma de entrada/salida
        # (las columnas enteras se declaran como float para tolerar faltantes al inferir)
        X_sig = X_train.astype({c: "float64" for c in X_train.select_dtypes("integer").columns})
        signature = infer_signature(X_sig, pipeline.predict(X_train))
        mlflow.sklearn.log_model(pipeline, name="modelo", signature=signature,
                                 input_example=X_sig.head(3))

    print(f"\nTracking URI : {uri}")
    print(f"Experimento  : {args.experiment_name}")
    print(f"Run          : {run_name} ({run.info.run_id})")
    for key, value in metrics.items():
        print(f"  test_{key:<10}: {value:.4f}")
    print(f"  cv_roc_auc   : {cv_auc.mean():.4f} ± {cv_auc.std():.4f}")
    return metrics


if __name__ == "__main__":
    main()
