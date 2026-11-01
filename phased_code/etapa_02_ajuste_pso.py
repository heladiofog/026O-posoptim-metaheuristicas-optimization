"""ETAPA 2. Selección de hiperparámetros PSO-TVAC: 12 configuraciones.

Mantiene los 4 pares (c1,c2) x 3 opciones (inercia,vmax) del código original,
8 semillas de calibración y criterio: menor media, desempate mediana/desv.
"""

from etapa_01_aleatorios import cargar_banco
from etapa_06_nucleo import (
    TUNE_SEEDS, OUT, pso_tvac, resumir_ajuste, ordenar_ajuste,
    escribir_csv, guardar_json,
)


def candidatos_pso():
    # c1 decreciente, c2 creciente
    rampas = [
        (2.5, 0.5, 0.5, 2.5),
        (2.8, 0.8, 0.8, 2.8),
        (2.4, 0.7, 0.7, 2.4),
        (2.0, 0.5, 0.5, 2.0),
    ]
    # c1 creciente, c2 decreciente
    # rampas = [
    #         (0.5, 2.5, 2.5, 0.5),
    #         (0.8, 2.8, 2.8, 0.8),
    #         (0.7, 2.4, 2.4, 0.7),
    #         (0.5, 2.0, 2.0, 0.5),
    #     ]
    opciones_inercia = [
        (0.90, 0.40, 0.20),
        (0.90, 0.40, 0.40),
        (0.80, 0.30, 0.30),
    ]
    return [
        {"c10": a, "c11": b, "c20": c, "c21": d,
         "w0": w0, "w1": w1, "vfrac": fraccion}
        for a, b, c, d in rampas
        for w0, w1, fraccion in opciones_inercia
    ]


def ajustar_pso():
    tabla = []
    for numero, parametros in enumerate(candidatos_pso(), 1):
        finales = []
        for semilla in TUNE_SEEDS:
            banco = cargar_banco(semilla)
            corrida = pso_tvac(banco["x0"], parametros, banco)
            finales.append(float(corrida["mejor"]))
        tabla.append(resumir_ajuste(numero, parametros, finales))
    return ordenar_ajuste(tabla)


def main():
    tabla = ajustar_pso()
    mejor = {clave: tabla[0][clave] for clave in
             ("c10", "c11", "c20", "c21", "w0", "w1", "vfrac")}
    OUT.mkdir(parents=True, exist_ok=True)
    escribir_csv(OUT / "ajuste_PSO.csv", tabla)
    guardar_json(OUT / "mejores_parametros_PSO.json", mejor)
    print(f"[Etapa 2] Configuraciones PSO evaluadas: {len(tabla)}")
    print(f"[Etapa 2] Mejor PSO = {mejor}; media = {tabla[0]['media']:.10f}")


if __name__ == "__main__":
    main()
