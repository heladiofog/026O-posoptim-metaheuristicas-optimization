"""ETAPA 1. Generación exclusiva de TODA la aleatoriedad.

Cada semilla crea un banco que se reutiliza SIN volver a sortear durante el
ajuste ni durante la comparación. En el código original los sorteos se hacen
DENTRO de las funciones de optimización; aquí son completamente auditables.

La estructura preserva exactamente los generadores y el orden de los sorteos
originales: semilla de población, semilla+100000 para PSO y +200000 para DE.
"""
from pathlib import Path

import numpy as np

from etapa_06_nucleo import (
    N, D, T, LOW, HIGH, BASE_SEED, TUNE_SEEDS, TEST_SEEDS,
    OUT, ALEATORIOS, escribir_csv, guardar_json,
)


def crear_banco(semilla):
    """Simula una sola vez las llamadas a RNG de ambos algoritmos.

    velocidad_uniforme U(0,1) permite reconstruir EXACTAMENTE una velocidad
    inicial de uniforme(-vmax,vmax) para cualquier vmax del ajuste PSO.
    """
    poblacion = np.random.default_rng(semilla).uniform(LOW, HIGH, size=(N, D))

    rng_pso = np.random.default_rng(semilla + 100000)
    velocidad_uniforme = rng_pso.random((N, D))
    pso_r1 = np.empty((T, N, D))
    pso_r2 = np.empty((T, N, D))
    for t in range(T):
        pso_r1[t] = rng_pso.random((N, D))
        pso_r2[t] = rng_pso.random((N, D))

    rng_de = np.random.default_rng(semilla + 200000)
    de_r1idx = np.empty((T, N), dtype=np.int16)
    de_r2idx = np.empty((T, N), dtype=np.int16)
    de_azar = np.empty((T, N, D))
    de_jrand = np.empty((T, N), dtype=np.int8)
    for t in range(T):
        for i in range(N):
            elegibles = np.delete(np.arange(N), i)
            idx1, idx2 = rng_de.choice(elegibles, size=2, replace=False)
            de_r1idx[t, i] = idx1
            de_r2idx[t, i] = idx2
            de_azar[t, i] = rng_de.random(D)
            de_jrand[t, i] = rng_de.integers(0, D)

    return {
        "semilla": np.array(semilla), "x0": poblacion,
        "velocidad_uniforme": velocidad_uniforme,
        "pso_r1": pso_r1, "pso_r2": pso_r2,
        "de_r1idx": de_r1idx, "de_r2idx": de_r2idx,
        "de_azar": de_azar, "de_jrand": de_jrand,
    }


def ruta_banco(semilla):
    return ALEATORIOS / f"banco_{semilla}.npz"


def guardar_banco(semilla):
    ALEATORIOS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(ruta_banco(semilla), **crear_banco(semilla))


def cargar_banco(semilla):
    """Exige pasar por etapa 1; nunca sortea silenciosamente dentro del ajuste."""
    ruta = ruta_banco(semilla)
    if not ruta.exists():
        raise FileNotFoundError(
            f"Falta {ruta.name}. Ejecute primero: python etapa_01_aleatorios.py")
    with np.load(ruta, allow_pickle=False) as fichero:
        banco = {clave: fichero[clave] for clave in fichero.files}
    if int(banco["semilla"]) != semilla:
        raise ValueError("La semilla del banco no coincide con el nombre de archivo")
    return banco


def exportar_csv_aleatorios_principales(banco):
    """CSV auxiliares para inspección humana (los .npz conservan todos los bancos)."""
    x0 = banco["x0"]
    escribir_csv(ALEATORIOS / "01_poblacion_inicial.csv", [
        {"individuo": i + 1, **{f"x{d+1}": x0[i, d] for d in range(D)}}
        for i in range(N)
    ])
    pso = []
    de = []
    for t in range(T):
        for i in range(N):
            pso.append({
                "iteracion": t+1, "individuo": i+1,
                **{f"r1_{d+1}": banco["pso_r1"][t, i, d] for d in range(D)},
                **{f"r2_{d+1}": banco["pso_r2"][t, i, d] for d in range(D)},
            })
            de.append({
                "generacion": t+1, "individuo": i+1,
                "indice_r1": int(banco["de_r1idx"][t, i]) + 1,
                "indice_r2": int(banco["de_r2idx"][t, i]) + 1,
                "jrand": int(banco["de_jrand"][t, i]) + 1,
                **{f"azar_{d+1}": banco["de_azar"][t, i, d] for d in range(D)},
            })
    escribir_csv(ALEATORIOS / "02_aleatorios_PSO_30x30.csv", pso)
    escribir_csv(ALEATORIOS / "03_aleatorios_DE_30x30.csv", de)
    escribir_csv(ALEATORIOS / "04_uniformes_velocidad_PSO.csv", [
        {"individuo": i + 1, **{
            f"U_v0_{d+1}": banco["velocidad_uniforme"][i, d] for d in range(D)
        }} for i in range(N)
    ])


def main():
    semillas = [BASE_SEED, *TUNE_SEEDS, *TEST_SEEDS]
    for semilla in semillas:
        guardar_banco(semilla)
    exportar_csv_aleatorios_principales(cargar_banco(BASE_SEED))
    guardar_json(ALEATORIOS / "manifest_aleatorios.json", {
        "funcion": "Ackley 3D",
        "semilla_principal": BASE_SEED,
        "semillas_calibracion": list(TUNE_SEEDS),
        "semillas_validacion": list(TEST_SEEDS),
        "bancos_generados": len(semillas),
        "nota": "PSO: generador semilla+100000; DE: semilla+200000; U_velocidad transforma a [-vmax,vmax].",
        "metodo": "Generadores NumPy PCG64, como default_rng del código original",
    })
    print(f"[Etapa 1] {len(semillas)} bancos .npz + 4 CSV auditables en: {ALEATORIOS}")


if __name__ == "__main__":
    main()
