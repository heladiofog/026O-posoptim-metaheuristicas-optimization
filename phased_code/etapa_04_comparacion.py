"""ETAPA 4. Comparación con mejores parámetros obtenidos en etapas 2 y 3.

Una corrida principal con semilla 314159 y 20 replicaciones pareadas con
semillas 2101..2120. Ambos algoritmos usan la MISMA población por semilla.
Genera métricas, trazas de 900 pasos/algoritmo y poblaciones para etapa 5.
"""
from statistics import mean

import numpy as np

from etapa_01_aleatorios import cargar_banco
from etapa_06_nucleo import (
    BASE_SEED, TEST_SEEDS, T, N, OUT,
    pso_tvac, de_current_to_best, escribir_csv, cargar_json, guardar_json,
    convertir_fila_pso, convertir_fila_de, tabla_evolucion, verificar_resultado,
)


def comparar_algoritmos():
    p_pso = cargar_json(OUT / "mejores_parametros_PSO.json")
    p_de = cargar_json(OUT / "mejores_parametros_DE.json")
    principal = cargar_banco(BASE_SEED)

    pso = pso_tvac(principal["x0"], p_pso, principal, guardar=True)
    de = de_current_to_best(principal["x0"], p_de, principal, guardar=True)
    verificar_resultado(pso, "PSO")
    verificar_resultado(de, "DE")

    escribir_csv(OUT / "PSO_pasos.csv", [convertir_fila_pso(p) for p in pso["pasos"]])
    escribir_csv(OUT / "DE_pasos.csv", [convertir_fila_de(p) for p in de["pasos"]])
    escribir_csv(OUT / "PSO_evolucion.csv", tabla_evolucion(pso))
    escribir_csv(OUT / "DE_evolucion.csv", tabla_evolucion(de))

    # (31,30,3), esenciales para construir boxplots sin repetir simulación.
    np.savez_compressed(OUT / "poblaciones_PSO.npz",
                        x=np.stack([e["coordenadas"] for e in pso["evolucion"]]))
    np.savez_compressed(OUT / "poblaciones_DE.npz",
                        x=np.stack([e["coordenadas"] for e in de["evolucion"]]))

    replicas = []
    for semilla in TEST_SEEDS:
        banco = cargar_banco(semilla)
        corrida_pso = pso_tvac(banco["x0"], p_pso, banco)
        corrida_de = de_current_to_best(banco["x0"], p_de, banco)
        verificar_resultado(corrida_pso, "PSO réplicas")
        verificar_resultado(corrida_de, "DE réplicas")
        replicas.append({
            "semilla": semilla,
            "PSO": float(corrida_pso["mejor"]),
            "DE": float(corrida_de["mejor"]),
            "PSO_gana": int(corrida_pso["mejor"] < corrida_de["mejor"]),
        })
    escribir_csv(OUT / "comparacion_20_repeticiones.csv", replicas)
    informe = {
        "funcion": "Ackley 3D", "optimo_teorico": {"x": [0, 0, 0], "f": 0},
        "dominio": [-10, 10], "poblacion": N, "iteraciones": T,
        "semilla_principal": BASE_SEED, "semillas_validacion": list(TEST_SEEDS),
        "pso_parametros": p_pso, "de_parametros": p_de,
        "pso_final": float(pso["mejor"]), "pso_x": pso["mejor_x"].tolist(),
        "de_final": float(de["mejor"]), "de_x": de["mejor_x"].tolist(),
        "pso_radio_final": float(pso["evolucion"][-1]["radio"]),
        "de_radio_final": float(de["evolucion"][-1]["radio"]),
        "pso_validacion_media": mean(r["PSO"] for r in replicas),
        "de_validacion_media": mean(r["DE"] for r in replicas),
        "pso_victorias": sum(r["PSO_gana"] for r in replicas),
        "de_victorias": sum(r["PSO"] > r["DE"] for r in replicas),
        "empates": sum(r["PSO"] == r["DE"] for r in replicas),
        "nota": "Los dos algoritmos optimizan variables reales mediante generaciones discretas."
    }
    guardar_json(OUT / "resultados_comparacion.json", informe)
    return informe


def main():
    informe = comparar_algoritmos()
    print("[Etapa 4] PSO final:", f'{informe["pso_final"]:.12f}',
          "| DE final:", f'{informe["de_final"]:.12f}')
    print("[Etapa 4] Promedios (20):",
          f'PSO={informe["pso_validacion_media"]:.12f}',
          f'DE={informe["de_validacion_media"]:.12f}')
    print("[Etapa 4] Victorias:", informe["pso_victorias"], "PSO /",
          informe["de_victorias"], "DE")


if __name__ == "__main__":
    main()
