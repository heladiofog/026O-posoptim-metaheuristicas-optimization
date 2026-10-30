"""ETAPA 6. Núcleo matemático, algoritmos, métricas, exportación y validación.

Corresponde al código original del Equipo 3, función Ackley (tres variables).
Esta etapa es una BIBLIOTECA compartida: las etapas 1–5 importan sus funciones.

Nota: PSO-TVAC y DE son métodos para variables continuas, pero avanzan por
iteraciones DISCRETAS. No se plantea una ecuación diferencial en tiempo continuo.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

# Reproduce las constantes y semillas del programa proporcionado.
D = 3
N = 30
T = 30
LOW, HIGH = -10.0, 10.0
BASE_SEED = 314159
TUNE_SEEDS = tuple(range(1101, 1109))    # 8 corridas de calibración.
TEST_SEEDS = tuple(range(2101, 2121))   # 20 corridas de comparación.
ROOT = Path(__file__).resolve().parent
OUT = ROOT / "resultados"
ALEATORIOS = OUT / "aleatorios"


def ackley(x):
    """Ackley tridimensional (vectoriza sobre la última dimensión)."""
    a = np.asarray(x, dtype=float)
    prom_cuadrados = np.mean(a * a, axis=-1)
    prom_cosenos = np.mean(np.cos(2.0 * np.pi * a), axis=-1)
    return (20.0 + np.e
            - 20.0 * np.exp(-0.2 * np.sqrt(prom_cuadrados))
            - np.exp(prom_cosenos))


def metricas(x, f, historico):
    """Mejor actual, histórico y radio ζ=max(||x_j - μ||_2)."""
    media = np.mean(x, axis=0)
    distancias = np.linalg.norm(x - media, axis=1)
    indice = int(np.argmin(f))
    return {
        "actual": float(f[indice]),
        "historico": float(historico),
        "mejor_x": x[indice].copy(),     # Mejor ACTUAL (no gbest histórico).
        "mu": media.copy(),
        "radio": float(np.max(distancias)),
        "coordenadas": x.copy(),
    }


def _validar_banco(banco):
    """Detecta archivos incompletos antes de efectuar la optimización."""
    dimensiones = {
        "x0": (N, D),
        "velocidad_uniforme": (N, D),
        "pso_r1": (T, N, D),
        "pso_r2": (T, N, D),
        "de_r1idx": (T, N),
        "de_r2idx": (T, N),
        "de_azar": (T, N, D),
        "de_jrand": (T, N),
    }
    for nombre, forma in dimensiones.items():
        if nombre not in banco or banco[nombre].shape != forma:
            raise ValueError(f"Banco aleatorio mal formado: {nombre}; esperado {forma}")
    if not np.all((banco["x0"] >= LOW) & (banco["x0"] <= HIGH)):
        raise ValueError("La población inicial sale del dominio")


def pso_tvac(x0, parametros, banco, guardar=False):
    """PSO-TVAC del código original con aleatorios pre-generados en etapa 1.

    La secuencia v0/r1/r2 preserva el orden de llamadas a NumPy del original.
    w, c1 y c2 cambian linealmente de t=1 a t=T.
    """
    _validar_banco(banco)
    x = np.array(x0, dtype=float, copy=True)
    vmax = float(parametros["vfrac"]) * (HIGH - LOW)
    # Coincide con rng.uniform(-vmax,vmax): -vmax + (2*vmax)*U.
    v = -vmax + (2.0 * vmax) * banco["velocidad_uniforme"]
    x_initial, v_initial = x.copy(), v.copy()
    pbest = x.copy()
    fpbest = ackley(x)
    indice = int(np.argmin(fpbest))
    gbest = pbest[indice].copy()
    fgbest = float(fpbest[indice])
    evolucion = [dict(iteracion=0, **metricas(x, fpbest, fgbest))]
    pasos = []

    for t in range(1, T + 1):
        progreso = (t - 1) / (T - 1)
        w = parametros["w0"] + progreso * (parametros["w1"] - parametros["w0"])
        c1 = parametros["c10"] + progreso * (parametros["c11"] - parametros["c10"])
        c2 = parametros["c20"] + progreso * (parametros["c21"] - parametros["c20"])
        r1, r2 = banco["pso_r1"][t - 1], banco["pso_r2"][t - 1]
        x_ant, v_ant, p_ant = x.copy(), v.copy(), pbest.copy()
        fp_ant, g_ant, fg_ant = fpbest.copy(), gbest.copy(), fgbest
        inercia = w * v_ant
        cognitivo = c1 * r1 * (p_ant - x_ant)
        social = c2 * r2 * (g_ant - x_ant)
        v_raw = inercia + cognitivo + social
        v = np.clip(v_raw, -vmax, vmax)
        x_raw = x_ant + v
        x = np.clip(x_raw, LOW, HIGH)
        f = ackley(x)
        mejora = f < fpbest
        pbest[mejora], fpbest[mejora] = x[mejora], f[mejora]
        mejoridx = int(np.argmin(fpbest))
        if float(fpbest[mejoridx]) < fgbest:
            fgbest = float(fpbest[mejoridx])
            gbest = pbest[mejoridx].copy()
        evolucion.append(dict(iteracion=t, **metricas(x, f, fgbest)))
        if guardar:
            for i in range(N):
                pasos.append({
                    "iteracion": t, "individuo": i + 1,
                    "x_ant": x_ant[i], "v_ant": v_ant[i], "p_ant": p_ant[i],
                    "fp_ant": float(fp_ant[i]), "g_ant": g_ant, "fg_ant": fg_ant,
                    "w": float(w), "c1": float(c1), "c2": float(c2),
                    "r1": r1[i], "r2": r2[i],
                    "inercia": inercia[i], "cognitivo": cognitivo[i], "social": social[i],
                    "v_raw": v_raw[i], "v": v[i], "x_raw": x_raw[i], "x": x[i],
                    "f": float(f[i]), "mejora": bool(mejora[i]),
                    "p": pbest[i].copy(), "fp": float(fpbest[i]),
                })
    return {
        "evolucion": evolucion, "pasos": pasos, "x0": x_initial,
        "v0": v_initial, "f0": ackley(x_initial),
        "mejor": fgbest, "mejor_x": gbest,
    }


def de_current_to_best(x0, parametros, banco, guardar=False):
    """DE/current-to-best/1/bin con elección de índices y cruce del banco 1."""
    _validar_banco(banco)
    x = np.array(x0, dtype=float, copy=True)
    f = ackley(x)
    historico = float(np.min(f))
    evolucion = [dict(iteracion=0, **metricas(x, f, historico))]
    pasos = []
    for t in range(1, T + 1):
        x_ant, f_ant = x.copy(), f.copy()
        indmejor = int(np.argmin(f_ant))
        mejor_ant = x_ant[indmejor].copy()
        x_nueva, f_nueva = x_ant.copy(), f_ant.copy()
        for i in range(N):
            idx1 = int(banco["de_r1idx"][t - 1, i])
            idx2 = int(banco["de_r2idx"][t - 1, i])
            if idx1 == idx2 or i in (idx1, idx2):
                raise ValueError("Índices DE inválidos")
            azar = banco["de_azar"][t - 1, i]
            jrand = int(banco["de_jrand"][t - 1, i])
            donante_raw = (x_ant[i]
                           + parametros["F"] * (mejor_ant - x_ant[i])
                           + parametros["F"] * (x_ant[idx1] - x_ant[idx2]))
            donante = np.clip(donante_raw, LOW, HIGH)
            mascara = (azar <= parametros["CR"]) | (np.arange(D) == jrand)
            prueba = np.where(mascara, donante, x_ant[i])
            fprueba = float(ackley(prueba))
            acepta = fprueba <= f_ant[i]
            if acepta:
                x_nueva[i], f_nueva[i] = prueba, fprueba
            if guardar:
                pasos.append({
                    "iteracion": t, "individuo": i + 1,
                    "x_ant": x_ant[i].copy(), "f_ant": float(f_ant[i]),
                    "best_ant": mejor_ant.copy(), "idx_best": indmejor + 1,
                    "r1idx": idx1 + 1, "r2idx": idx2 + 1,
                    "xr1": x_ant[idx1].copy(), "xr2": x_ant[idx2].copy(),
                    "F": float(parametros["F"]), "CR": float(parametros["CR"]),
                    "azar": azar.copy(), "jrand": jrand + 1,
                    "donante_raw": donante_raw, "donante": donante,
                    "mascara": mascara.copy(), "prueba": prueba.copy(),
                    "fprueba": fprueba, "acepta": bool(acepta),
                    "x": x_nueva[i].copy(), "f": float(f_nueva[i]),
                })
        x, f = x_nueva, f_nueva
        historico = min(historico, float(np.min(f)))
        evolucion.append(dict(iteracion=t, **metricas(x, f, historico)))
    indice = int(np.argmin(f))
    return {
        "evolucion": evolucion, "pasos": pasos, "x0": np.array(x0, copy=True),
        "f0": ackley(x0), "mejor": historico,
        "mejor_x": x[indice].copy(),
    }


def resumir_ajuste(numero, parametros, finales):
    """Reglas estadísticas de ajuste conservadas del original."""
    a = np.asarray(finales, dtype=float)
    return {
        "id": numero, **parametros,
        "media": float(np.mean(a)), "mediana": float(np.median(a)),
        "desv": float(np.std(a, ddof=1)),
        "minimo": float(np.min(a)), "maximo": float(np.max(a)),
        "exitos_001": int(np.sum(a <= 0.01)),
        "evaluaciones": N * T,   # Campo original: sólo las 900 actualizaciones.
        "evaluaciones_con_inicial": N * (T + 1),
        "repeticiones": len(a),
    }


def ordenar_ajuste(tabla):
    tabla.sort(key=lambda fila: (fila["media"], fila["mediana"], fila["desv"]))
    for puesto, fila in enumerate(tabla, start=1):
        fila["puesto"] = puesto
    return tabla


def escribir_csv(ruta, filas):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if not filas:
        raise ValueError(f"No existen registros que exportar a {ruta}")
    with ruta.open("w", encoding="utf-8-sig", newline="") as archivo:
        escritor = csv.DictWriter(archivo, fieldnames=list(filas[0].keys()))
        escritor.writeheader()
        escritor.writerows(filas)


def guardar_json(ruta, datos):
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    ruta.write_text(json.dumps(datos, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def cargar_json(ruta):
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def convertir_fila_pso(p):
    fila = {k: p[k] for k in (
        "iteracion", "individuo", "fp_ant", "fg_ant", "w", "c1", "c2", "f", "fp")}
    fila["mejora"] = int(p["mejora"])
    for clave in ("x_ant", "v_ant", "p_ant", "g_ant", "r1", "r2", "inercia",
                  "cognitivo", "social", "v_raw", "v", "x_raw", "x", "p"):
        for d in range(D):
            fila[f"{clave}_{d+1}"] = float(p[clave][d])
    return fila


def convertir_fila_de(p):
    fila = {clave: p[clave] for clave in (
        "iteracion", "individuo", "f_ant", "idx_best", "r1idx", "r2idx",
        "F", "CR", "jrand", "fprueba", "f")}
    fila["acepta"] = int(p["acepta"])
    for clave in ("x_ant", "best_ant", "xr1", "xr2", "azar", "donante_raw",
                  "donante", "prueba", "x"):
        for d in range(D):
            fila[f"{clave}_{d+1}"] = float(p[clave][d])
            if clave == "prueba":
                fila[f"cruce_{d+1}"] = int(p["mascara"][d])
    return fila


def tabla_evolucion(resultado):
    return [{
        "iteracion": e["iteracion"],
        "mejor_poblacion": e["actual"], "mejor_historico": e["historico"],
        "mu1": float(e["mu"][0]), "mu2": float(e["mu"][1]),
        "mu3": float(e["mu"][2]), "radio": e["radio"],
        "mejor_x1": float(e["mejor_x"][0]),
        "mejor_x2": float(e["mejor_x"][1]),
        "mejor_x3": float(e["mejor_x"][2]),
    } for e in resultado["evolucion"]]


def verificar_resultado(resultado, nombre):
    """Pruebas numéricas independientes de la trayectoria y sus restricciones."""
    ev = resultado["evolucion"]
    if len(ev) != T + 1:
        raise AssertionError(f"{nombre}: número de iteraciones incorrecto")
    historicos = []
    for iteracion, e in enumerate(ev):
        x = np.asarray(e["coordenadas"])
        if x.shape != (N, D) or not np.all(np.isfinite(x)):
            raise AssertionError(f"{nombre}: población inválida en t={iteracion}")
        if np.min(x) < LOW - 1e-12 or np.max(x) > HIGH + 1e-12:
            raise AssertionError(f"{nombre}: individuo fuera del dominio")
        f = ackley(x)
        radio = float(np.max(np.linalg.norm(x - np.mean(x, axis=0), axis=1)))
        if not np.isclose(np.min(f), e["actual"], atol=1e-12):
            raise AssertionError(f"{nombre}: valor actual incoherente")
        if not np.isclose(radio, e["radio"], atol=1e-12):
            raise AssertionError(f"{nombre}: radio mal calculado")
        historicos.append(e["historico"])
    if np.any(np.diff(historicos) > 1e-12):
        raise AssertionError(f"{nombre}: mejor histórico aumenta")
    if not np.isclose(resultado["mejor"], historicos[-1], atol=1e-12):
        raise AssertionError(f"{nombre}: resultado final inconsistente")
    if resultado["pasos"] and len(resultado["pasos"]) != N * T:
        raise AssertionError(f"{nombre}: traza incompleta")


def pruebas_basicas():
    assert abs(float(ackley(np.zeros(D)))) < 1e-12
    assert ackley(np.zeros((5, D))).shape == (5,)
    prueba = metricas(np.zeros((N, D)), np.zeros(N), 0.0)
    assert prueba["radio"] == 0.0
    print("[Etapa 6] Fórmula de Ackley y métricas básicas: correctas.")


if __name__ == "__main__":
    pruebas_basicas()
