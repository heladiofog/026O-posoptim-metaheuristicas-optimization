"""ETAPA 3. Selección de parámetros DE con EXACTAMENTE 9 combinaciones.

F={0.30, 0.45, 0.60} y CR={0.50, 0.75, 0.90}; 3x3=9.
Conserva el par anteriormente ganador (F=0.45, CR=0.90) y excluye F=0.75
(que sí figuraba entre las 12 opciones del código original).
"""

from etapa_01_aleatorios import cargar_banco
from etapa_06_nucleo import (
    TUNE_SEEDS, OUT, de_current_to_best, resumir_ajuste,
    ordenar_ajuste, escribir_csv, guardar_json,
)


def candidatos_de():
    valores_f = (0.30, 0.45, 0.60)
    valores_cr = (0.50, 0.75, 0.90)
    parametros = [{"F": f, "CR": cr} for f in valores_f for cr in valores_cr]
    assert len(parametros) == 9
    assert {"F": 0.45, "CR": 0.90} in parametros
    return parametros


def ajustar_de():
    tabla = []
    for numero, parametros in enumerate(candidatos_de(), 1):
        finales = []
        for semilla in TUNE_SEEDS:
            banco = cargar_banco(semilla)
            corrida = de_current_to_best(banco["x0"], parametros, banco)
            finales.append(float(corrida["mejor"]))
        tabla.append(resumir_ajuste(numero, parametros, finales))
    return ordenar_ajuste(tabla)


def main():
    tabla = ajustar_de()
    mejor = {clave: tabla[0][clave] for clave in ("F", "CR")}
    OUT.mkdir(parents=True, exist_ok=True)
    escribir_csv(OUT / "ajuste_DE_9_combinaciones.csv", tabla)
    guardar_json(OUT / "mejores_parametros_DE.json", mejor)
    print(f"[Etapa 3] Combinaciones DE evaluadas: {len(tabla)}")
    print(f"[Etapa 3] Mejor DE = {mejor}; media = {tabla[0]['media']:.10f}")


if __name__ == "__main__":
    main()
