# TC5061 · Experimentación base con MLflow — Telco Customer Churn

Script de entrenamiento parametrizado e instrumentado con **MLflow Tracking** para predecir el abandono de clientes (*churn*) de una empresa de telecomunicaciones. El modelo es una regresión logística dentro de un `Pipeline` de scikit-learn (escalado + one-hot encoding + clasificador).

**Autora:** Daniela Montserrat Enríquez Ballesteros · Maestría en IA Aplicada, Tec de Monterrey

## Estructura

```
├── train.py               # Carga, limpieza, entrenamiento, evaluación y registro en MLflow
├── run_experiments.py     # Ejecuta las 8 corridas del experimento llamando a train.py
├── requirements.txt       # Dependencias con versiones fijas
└── notebooks/
    └── 01_EDA_Telco_Churn.ipynb   # Solo exploración y decisiones de limpieza
```

El dataset ([Telco Customer Churn, IBM](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)) se descarga automáticamente en `data/` la primera vez que se ejecuta el script.

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución

1. Levantar el servidor de MLflow en una terminal y dejarla abierta:

   ```bash
   mlflow server --host 127.0.0.1 --port 5000
   ```

2. En otra terminal, entrenar con los valores por defecto:

   ```bash
   python train.py
   ```

3. Cambiar hiperparámetros sin editar el código:

   ```bash
   python train.py --C 0.1 --penalty l1 --class-weight balanced --seed 42
   ```

4. Ejecutar todas las corridas del experimento:

   ```bash
   python run_experiments.py
   ```

5. Abrir `http://127.0.0.1:5000`, entrar al experimento **TC5061_Telco_Churn_LogReg** y comparar los runs.

Si el servidor no está disponible, el script registra en un archivo local `mlflow.db`, que se puede consultar con `mlflow ui --backend-store-uri sqlite:///mlflow.db`.

## Argumentos

| Argumento | Default | Descripción |
|---|---|---|
| `--C` | `1.0` | Inverso de la fuerza de regularización |
| `--penalty` | `l2` | Tipo de regularización (`l1`, `l2`) |
| `--class-weight` | `none` | `balanced` pondera las clases según su frecuencia |
| `--max-iter` | `1000` | Iteraciones máximas del optimizador |
| `--threshold` | `0.5` | Umbral de probabilidad para clasificar como churn |
| `--test-size` | `0.2` | Proporción del conjunto de prueba |
| `--seed` | `42` | Semilla aleatoria global |
| `--experiment-name` | `TC5061_Telco_Churn_LogReg` | Experimento de MLflow |
| `--run-name` | automático | Nombre del run |
| `--tracking-uri` | `http://127.0.0.1:5000` | Servidor de MLflow (o variable `MLFLOW_TRACKING_URI`) |

`python train.py --help` muestra la lista completa.

## Qué se registra en cada run

- **Parámetros** (`log_param`): hiperparámetros, solver, semilla, tamaños de los conjuntos y el hash MD5 del CSV (para detectar si los datos cambiaron).
- **Métricas** (`log_metric`): accuracy, precision, recall, F1, ROC-AUC y PR-AUC en prueba, más el ROC-AUC promedio y la desviación de una validación cruzada de 5 folds.
- **Artefactos** (`log_artifact`): matriz de confusión, curva ROC, reporte de clasificación y `metrics.json`.
- **Modelo** (`mlflow.sklearn.log_model`): el pipeline completo con su firma y un ejemplo de entrada.

## Reproducibilidad

- La semilla (`--seed`) se fija en Python y NumPy y se pasa a la partición estratificada, a la validación cruzada y al modelo.
- El solver `liblinear` es determinista para una misma semilla y datos.
- `run_experiments.py` ejecuta dos veces la configuración base (`01_baseline` y `02_baseline_repetido`): ambas corridas producen métricas idénticas.
- Las versiones de las librerías están fijadas en `requirements.txt`.
