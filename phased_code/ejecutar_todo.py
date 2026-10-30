"""Ejecuta todas las etapas en orden. También pueden ejecutarse por separado."""
import subprocess
import sys
from pathlib import Path

RUTA = Path(__file__).resolve().parent
ETAPAS = [
    "etapa_01_aleatorios.py",
    "etapa_02_ajuste_pso.py",
    "etapa_03_ajuste_de.py",
    "etapa_04_comparacion.py",
    "etapa_05_graficas.py",
]

if __name__ == "__main__":
    for etapa in ETAPAS:
        subprocess.run([sys.executable, str(RUTA / etapa)], cwd=RUTA, check=True)
    print("\nProceso completo. Consulte la carpeta resultados/.")
