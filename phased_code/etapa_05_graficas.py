"""ETAPA 5. Gráficas del cuestionario desde archivos de la etapa 4.

No vuelve a ejecutar algoritmos ni a sortear números aleatorios.
(a) mejor histórico; (b) mejor de población; (c) cajas por dimensión;
(d) radio poblacional ζ=max_j ||x_j-μ||_2.
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from etapa_06_nucleo import D, T, OUT


def cargar_evolucion(nombre):
    ruta = OUT / f"{nombre}_evolucion.csv"
    with ruta.open(encoding="utf-8-sig", newline="") as fp:
        datos = list(csv.DictReader(fp))
    if len(datos) != T + 1:
        raise ValueError(f"Datos de {nombre} incompletos para graficar")
    return {col: np.array([float(f[col]) for f in datos])
            for col in ("iteracion", "mejor_historico", "mejor_poblacion", "radio")}


def cargar_poblacion(nombre):
    with np.load(OUT / f"poblaciones_{nombre}.npz", allow_pickle=False) as data:
        x = data["x"]
    if x.shape != (T + 1, 30, D):
        raise ValueError(f"Dimensiones incorrectas de población {nombre}: {x.shape}")
    return x


def graficar_convergencia(pso, de):
    t = np.arange(T + 1)
    fig, ejes = plt.subplots(2, 1, figsize=(11.5, 8.3), sharex=True,
                             constrained_layout=True)
    paneles = [
        ("mejor_historico", "(a) Mejor valor encontrado hasta la iteración"),
        ("mejor_poblacion", "(b) Mejor valor en la población de cada iteración"),
    ]
    for ax, (clave, titulo) in zip(ejes, paneles):
        ax.plot(t, pso[clave], "-o", linewidth=1.6, markersize=3.1, label="PSO-TVAC")
        ax.plot(t, de[clave], "-s", linewidth=1.6, markersize=3.0,
                label="DE/current-to-best/1/bin")
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_title(titulo, loc="left", fontweight="bold")
        ax.set_ylabel("Ackley f3(x) (escala simlog)")
        ax.grid(True, alpha=0.25)
        ax.legend()
    ejes[-1].set_xlabel("Iteración / generación")
    fig.suptitle("Equipo 3 | Ackley | 30 individuos | 30 iteraciones", weight="bold")
    fig.savefig(OUT / "01_convergencia.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def graficar_radio(pso, de):
    fig, ax = plt.subplots(figsize=(11.5, 4.9), constrained_layout=True)
    t = np.arange(T + 1)
    ax.plot(t, pso["radio"], "-o", markersize=3, label="PSO-TVAC")
    ax.plot(t, de["radio"], "-s", markersize=3,
            label="DE/current-to-best/1/bin")
    ax.set(xlabel="Iteración / generación",
           ylabel="Radio poblacional ζ (norma euclidiana)",
           title="(d) Radio poblacional en cada iteración")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.savefig(OUT / "02_radio_poblacional.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def graficar_cajas(poblaciones, nombre, archivo):
    fig, ejes = plt.subplots(D, 1, figsize=(15, 11), sharex=True,
                             constrained_layout=True)
    for d, ax in enumerate(ejes):
        muestras = [poblaciones[t, :, d] for t in range(1, T + 1)]
        ax.boxplot(muestras, positions=np.arange(1, T + 1), widths=0.55,
                   showfliers=True, manage_ticks=False)
        ax.set_ylabel(f"Dimensión x{d+1}")
        ax.grid(axis="y", alpha=0.22)
        ax.axhline(0, linestyle=":", linewidth=0.9)
    ejes[-1].set_xticks(np.arange(1, T + 1))
    ejes[-1].set_xlabel("Iteración / generación")
    fig.suptitle(f"(c) Cajas y bigotes | {nombre} | 3 dimensiones", weight="bold")
    fig.savefig(OUT / archivo, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main():
    pso, de = cargar_evolucion("PSO"), cargar_evolucion("DE")
    graficar_convergencia(pso, de)
    graficar_radio(pso, de)
    graficar_cajas(cargar_poblacion("PSO"), "PSO-TVAC", "03_boxplots_PSO.png")
    graficar_cajas(cargar_poblacion("DE"), "DE/current-to-best/1/bin", "04_boxplots_DE.png")
    print(f"[Etapa 5] Cuatro figuras generadas en {OUT}")


if __name__ == "__main__":
    main()
