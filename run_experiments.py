"""
run_experiments.py — Ejecuta la batería de corridas del experimento llamando a train.py
desde la línea de comandos, una vez por configuración.

Uso:
    python run_experiments.py
    python run_experiments.py --tracking-uri http://127.0.0.1:5000

La primera y la segunda corrida usan exactamente los mismos argumentos para
comprobar la reproducibilidad (deben producir métricas idénticas).
"""

import argparse
import subprocess
import sys

EXPERIMENT = "TC5061_Telco_Churn_LogReg"

# (nombre del run, argumentos de train.py)
CORRIDAS = [
    ("01_baseline",              []),
    ("02_baseline_repetido",     []),
    ("03_C0.01_fuerte_reg",      ["--C", "0.01"]),
    ("04_C100_debil_reg",        ["--C", "100"]),
    ("05_l1_C0.1",               ["--penalty", "l1", "--C", "0.1"]),
    ("06_balanced",              ["--class-weight", "balanced"]),
    ("07_l1_C0.1_balanced",      ["--penalty", "l1", "--C", "0.1", "--class-weight", "balanced"]),
    ("08_umbral_0.3",            ["--threshold", "0.3"]),
]


def main():
    p = argparse.ArgumentParser(description="Ejecuta todas las corridas del experimento.")
    p.add_argument("--tracking-uri", default=None, help="URI de MLflow (por defecto la de train.py).")
    args = p.parse_args()

    for run_name, extra in CORRIDAS:
        cmd = [sys.executable, "train.py", *extra,
               "--experiment-name", EXPERIMENT, "--run-name", run_name]
        if args.tracking_uri:
            cmd += ["--tracking-uri", args.tracking_uri]
        print("\n$ " + " ".join(cmd[1:]))
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
