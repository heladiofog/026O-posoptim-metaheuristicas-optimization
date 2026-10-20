# EQUIPO 3 | COMPARACIÓN REPRODUCIBLE DE PSO-TVAC Y DE/current-to-best/1/bin.
# Requisitos: Python 3.10+, numpy y matplotlib; instalar con: pip install numpy matplotlib.
# Ejecución: python Equipo3_Ackley_codigo_comentado.py.
# El programa crea resultados, trazas CSV, tablas de ajuste y cuatro figuras PNG.
import csv  # Escribe tablas auditables, que también pueden abrirse en Excel.
import json  # Conserva los parámetros ganadores y los resultados en un archivo legible.
from pathlib import Path  # Construye rutas portables para Windows, macOS y Linux.
import numpy as np  # Ejecuta cálculos vectorizados en poblaciones de 30 individuos.
import matplotlib.pyplot as plt  # Produce las gráficas solicitadas para la comparación.

D = 3  # Número de variables independientes de la función de Ackley.
N = 30  # Tamaño fijo de población para las dos metaheurísticas.
T = 30  # Número de iteraciones para cada algoritmo.
LOW = -10.0  # Cota inferior común de cada coordenada.
HIGH = 10.0  # Cota superior común de cada coordenada.
BASE_SEED = 314159  # Semilla del experimento principal reproducible.
TUNE_SEEDS = list(range(1101, 1109))  # Ocho semillas independientes para ajustar parámetros.
TEST_SEEDS = list(range(2101, 2121))  # Veinte semillas independientes para validar los resultados.
OUT = Path(__file__).resolve().parent / 'Equipo3_resultados'  # Carpeta automática de resultados.
OUT.mkdir(parents=True, exist_ok=True)  # Crea la carpeta solamente si no existe.


def ackley(x):  # Evalúa la función f3 en uno o varios vectores de tres dimensiones.
    a = np.asarray(x, dtype=float)  # Convierte la entrada en un arreglo numérico reproducible.
    promedio_cuadrados = np.mean(a * a, axis=-1)  # Calcula (x1²+x2²+x3²)/3.
    promedio_cosenos = np.mean(np.cos(2.0 * np.pi * a), axis=-1)  # Calcula la media de cos(2πxi).
    return 20.0 + np.e - 20.0 * np.exp(-0.2 * np.sqrt(promedio_cuadrados)) - np.exp(promedio_cosenos)  # Fórmula completa de Ackley.


def punto_inicial(semilla):  # Genera una población común para permitir una comparación justa.
    return np.random.default_rng(semilla).uniform(LOW, HIGH, size=(N, D))  # Muestrea x de [-10,10]³.


def metricas(x, f, historico):  # Resume la calidad y la distribución espacial de la población.
    media = np.mean(x, axis=0)  # Calcula μ = (Σx_j)/30, componente por componente.
    distancias = np.linalg.norm(x - media, axis=1)  # Calcula ||x_j - μ||₂ para los 30 individuos.
    indice = int(np.argmin(f))  # Localiza el individuo de menor valor objetivo actual.
    return {'actual': float(f[indice]), 'historico': float(historico), 'mejor_x': x[indice].copy(), 'mu': media.copy(), 'radio': float(np.max(distancias)), 'coordenadas': x.copy()}  # Retorna todos los indicadores.


def pso_tvac(x0, parametros, semilla, guardar=False):  # Ejecuta PSO con factores de aceleración que cambian con el tiempo.
    rng = np.random.default_rng(semilla)  # Usa un generador propio para cada corrida de PSO.
    x = np.array(x0, dtype=float, copy=True)  # Copia la población inicial sin modificar la de DE.
    vmax = parametros['vfrac'] * (HIGH - LOW)  # Obtiene el límite simétrico |v| ≤ vfrac*20.
    v = rng.uniform(-vmax, vmax, size=(N, D))  # Inicializa la velocidad de cada dimensión.
    x_initial = x.copy()  # Conserva las coordenadas iniciales para las tablas de Excel.
    v_initial = v.copy()  # Conserva las velocidades iniciales para las tablas de Excel.
    pbest = x.copy()  # Inicializa la mejor posición personal en su posición inicial.
    fpbest = ackley(x)  # Calcula el valor objetivo del mejor punto personal inicial.
    posmejor = int(np.argmin(fpbest))  # Busca el índice del mejor individuo inicial.
    gbest = pbest[posmejor].copy()  # Guarda la posición global inicial, antes de actualizar partículas.
    fgbest = float(fpbest[posmejor])  # Guarda el mejor valor global inicial.
    inicial = metricas(x, fpbest, fgbest)  # Calcula los indicadores en la iteración cero.
    evolucion = [dict(iteracion=0, **inicial)]  # Conserva una fila de evolución previa a las 30 iteraciones.
    pasos = []  # Crea el registro de términos intermedios por partícula e iteración.
    for t in range(1, T + 1):  # Ejecuta exactamente 30 actualizaciones completas de población.
        progreso = (t - 1) / (T - 1)  # Define el factor de interpolación lineal de TVAC entre 0 y 1.
        w = parametros['w0'] + progreso * (parametros['w1'] - parametros['w0'])  # Reduce la inercia con las iteraciones.
        c1 = parametros['c10'] + progreso * (parametros['c11'] - parametros['c10'])  # Cambia el peso cognitivo TVAC.
        c2 = parametros['c20'] + progreso * (parametros['c21'] - parametros['c20'])  # Cambia el peso social TVAC.
        r1 = rng.random(size=(N, D))  # Genera un aleatorio cognitivo por dimensión y partícula.
        r2 = rng.random(size=(N, D))  # Genera un aleatorio social por dimensión y partícula.
        x_ant = x.copy()  # Conserva x(t-1) antes de actualizar en bloque.
        v_ant = v.copy()  # Conserva v(t-1) antes de actualizar en bloque.
        p_ant = pbest.copy()  # Conserva pbest para auditar el término cognitivo.
        fp_ant = fpbest.copy()  # Conserva las aptitudes personales antes de la iteración.
        g_ant = gbest.copy()  # Conserva el gbest al comienzo de la iteración.
        fg_ant = fgbest  # Conserva el mejor valor histórico hasta la iteración anterior.
        inercia = w * v_ant  # Calcula w(t)*v(t-1) para cada coordenada.
        cognitivo = c1 * r1 * (p_ant - x_ant)  # Calcula c1(t)*r1*(pbest-x).
        social = c2 * r2 * (g_ant - x_ant)  # Calcula c2(t)*r2*(gbest-x).
        v_raw = inercia + cognitivo + social  # Suma los tres términos de la velocidad PSO.
        v = np.clip(v_raw, -vmax, vmax)  # Recorta la velocidad para impedir saltos excesivos.
        x_raw = x_ant + v  # Aplica x(t)=x(t-1)+v(t), antes del control de fronteras.
        x = np.clip(x_raw, LOW, HIGH)  # Restringe cada coordenada al cubo [-10,10]³.
        f = ackley(x)  # Evalúa Ackley en las 30 nuevas posiciones.
        mejora = f < fpbest  # Detecta si la nueva posición supera el mejor personal anterior.
        pbest[mejora] = x[mejora]  # Sustituye pbest exclusivamente si la solución mejoró.
        fpbest[mejora] = f[mejora]  # Actualiza el valor personal al mismo tiempo que pbest.
        mejoridx = int(np.argmin(fpbest))  # Localiza el mejor personal después de toda la iteración.
        if float(fpbest[mejoridx]) < fgbest:  # Comprueba si existe una mejora global estricta.
            fgbest = float(fpbest[mejoridx])  # Actualiza la mejor aptitud global encontrada.
            gbest = pbest[mejoridx].copy()  # Actualiza la posición de la mejor solución histórica.
        evolucion.append(dict(iteracion=t, **metricas(x, f, fgbest)))  # Guarda ambos mejores y el radio.
        if guardar:  # Evita almacenar tablas gigantes durante el ajuste de parámetros.
            for i in range(N):  # Agrega una fila de auditoría para cada partícula.
                pasos.append({'iteracion': t, 'individuo': i + 1, 'x_ant': x_ant[i], 'v_ant': v_ant[i], 'p_ant': p_ant[i], 'fp_ant': float(fp_ant[i]), 'g_ant': g_ant, 'fg_ant': fg_ant, 'w': float(w), 'c1': float(c1), 'c2': float(c2), 'r1': r1[i], 'r2': r2[i], 'inercia': inercia[i], 'cognitivo': cognitivo[i], 'social': social[i], 'v_raw': v_raw[i], 'v': v[i], 'x_raw': x_raw[i], 'x': x[i], 'f': float(f[i]), 'mejora': bool(mejora[i]), 'p': pbest[i].copy(), 'fp': float(fpbest[i])})  # Conserva todas las entradas y salidas del paso.
    return {'evolucion': evolucion, 'pasos': pasos, 'x0': x_initial, 'v0': v_initial, 'f0': ackley(x_initial), 'mejor': fgbest, 'mejor_x': gbest}  # Retorna los resultados de PSO.


def de_current_to_best(x0, parametros, semilla, guardar=False):  # Ejecuta la variante DE/current-to-best/1/bin.
    rng = np.random.default_rng(semilla)  # Genera los sorteos de mutación y cruce exclusivos de DE.
    x = np.array(x0, dtype=float, copy=True)  # Copia la población inicial de PSO para comparar justamente.
    f = ackley(x)  # Evalúa los 30 individuos de la generación inicial.
    historico = float(np.min(f))  # Inicializa el mejor valor registrado.
    evolucion = [dict(iteracion=0, **metricas(x, f, historico))]  # Guarda el diagnóstico en generación cero.
    pasos = []  # Prepara la bitácora para mutación, recombinación y selección.
    for t in range(1, T + 1):  # Ejecuta 30 generaciones completas de evolución diferencial.
        x_ant = x.copy()  # Congela la población objetivo de la generación anterior.
        f_ant = f.copy()  # Congela los valores objetivo para selección síncrona.
        indmejor = int(np.argmin(f_ant))  # Identifica al mejor individuo anterior.
        mejor_ant = x_ant[indmejor].copy()  # Conserva x_best utilizado en todas las mutaciones.
        x_nueva = x_ant.copy()  # Prepara los individuos elegidos tras selección.
        f_nueva = f_ant.copy()  # Prepara los valores objetivos de la población elegida.
        for i in range(N):  # Procesa cada vector objetivo sin modificar el origen de mutaciones.
            elegibles = np.delete(np.arange(N), i)  # Excluye el vector objetivo de los índices aleatorios.
            idx1, idx2 = rng.choice(elegibles, size=2, replace=False)  # Obtiene r1≠r2 y ambos distintos de i.
            azar = rng.random(D)  # Produce un número uniforme por componente para el cruce binomial.
            jrand = int(rng.integers(0, D))  # Fuerza al menos una dimensión donante en el vector de prueba.
            donante_raw = x_ant[i] + parametros['F'] * (mejor_ant - x_ant[i]) + parametros['F'] * (x_ant[idx1] - x_ant[idx2])  # Implementa current-to-best/1.
            donante = np.clip(donante_raw, LOW, HIGH)  # Restringe el donante a las cotas de dominio.
            mascara = (azar <= parametros['CR']) | (np.arange(D) == jrand)  # Aplica cruce binomial con dimensión forzada.
            prueba = np.where(mascara, donante, x_ant[i])  # Construye el vector de prueba u_i.
            fprueba = float(ackley(prueba))  # Evalúa la aptitud de la solución de prueba.
            acepta = fprueba <= f_ant[i]  # Acepta la prueba si no empeora a su vector objetivo.
            if acepta:  # Actualiza únicamente la población de la siguiente generación.
                x_nueva[i] = prueba  # Conserva el vector de prueba seleccionado.
                f_nueva[i] = fprueba  # Conserva el correspondiente valor de la función.
            if guardar:  # Registra mutación, cruce, condición de aceptación y selección.
                pasos.append({'iteracion': t, 'individuo': i + 1, 'x_ant': x_ant[i].copy(), 'f_ant': float(f_ant[i]), 'best_ant': mejor_ant.copy(), 'idx_best': indmejor + 1, 'r1idx': int(idx1) + 1, 'r2idx': int(idx2) + 1, 'xr1': x_ant[idx1].copy(), 'xr2': x_ant[idx2].copy(), 'F': float(parametros['F']), 'CR': float(parametros['CR']), 'azar': azar, 'jrand': jrand + 1, 'donante_raw': donante_raw, 'donante': donante, 'mascara': mascara.copy(), 'prueba': prueba.copy(), 'fprueba': fprueba, 'acepta': bool(acepta), 'x': x_nueva[i].copy(), 'f': float(f_nueva[i])})  # Agrega una fila completa al historial.
        x = x_nueva  # Realiza el reemplazo síncrono al final de la generación.
        f = f_nueva  # Actualiza en bloque las aptitudes de la generación.
        historico = min(historico, float(np.min(f)))  # Actualiza el mejor valor histórico, sin perder soluciones.
        evolucion.append(dict(iteracion=t, **metricas(x, f, historico)))  # Calcula media y radio de la nueva población.
    indice = int(np.argmin(f))  # Identifica la posición de la mejor solución final.
    return {'evolucion': evolucion, 'pasos': pasos, 'x0': np.array(x0, copy=True), 'f0': ackley(x0), 'mejor': historico, 'mejor_x': x[indice].copy()}  # Retorna la solución y el registro DE.


def candidatos_pso():  # Construye el espacio finito de 12 configuraciones TVAC para la búsqueda.
    cpares = [(2.5, 0.5, 0.5, 2.5), (2.8, 0.8, 0.8, 2.8), (2.4, 0.7, 0.7, 2.4), (2.0, 0.5, 0.5, 2.0)]  # Define cuatro rampas de aceleración.
    inercias = [(0.90, 0.40, 0.20), (0.90, 0.40, 0.40), (0.80, 0.30, 0.30)]  # Define tres parejas de inercia y velocidad máxima.
    return [dict(c10=a, c11=b, c20=c, c21=d, w0=w0, w1=w1, vfrac=frac) for a, b, c, d in cpares for w0, w1, frac in inercias]  # Combina 4×3 opciones.


def candidatos_de():  # Construye 12 combinaciones de escala diferencial y cruce.
    return [dict(F=F, CR=CR) for F in (0.30, 0.45, 0.60, 0.75) for CR in (0.50, 0.75, 0.90)]  # Barre cuatro F y tres CR.


def ajustar(nombre, candidatos, ejecutar):  # Compara parámetros con presupuestos y poblaciones iguales.
    tabla = []  # Guarda métricas de desempeño de cada configuración candidata.
    for num, parametros in enumerate(candidatos, start=1):  # Evalúa cada combinación de hiperparámetros.
        finales = []  # Recopila los ocho mejores valores alcanzados en las semillas de calibración.
        for semilla in TUNE_SEEDS:  # Repite el experimento con ocho poblaciones iniciales independientes.
            inicial = punto_inicial(semilla)  # Genera la misma población para cada algoritmo en cada semilla.
            corr = ejecutar(inicial, parametros, semilla + (100000 if nombre == 'PSO' else 200000))  # Ejecuta 30 iteraciones y 30 individuos.
            finales.append(float(corr['mejor']))  # Registra el mejor valor final de la corrida.
        finales = np.array(finales, dtype=float)  # Convierte a arreglo para calcular estadísticos.
        tabla.append({'id': num, **parametros, 'media': float(np.mean(finales)), 'mediana': float(np.median(finales)), 'desv': float(np.std(finales, ddof=1)), 'minimo': float(np.min(finales)), 'maximo': float(np.max(finales)), 'exitos_001': int(np.sum(finales <= 0.01)), 'evaluaciones': N * T, 'repeticiones': len(TUNE_SEEDS)})  # Retiene parámetros y métricas.
    tabla.sort(key=lambda f: (f['media'], f['mediana'], f['desv']))  # Selecciona por menor promedio; desempata con mediana y desviación.
    for pos, fila in enumerate(tabla, start=1):  # Rellena la clasificación global de configuraciones.
        fila['puesto'] = pos  # Expone el lugar que obtuvo cada parámetro.
    return tabla  # Entrega las configuraciones ordenadas de mejor a peor.


def escribir_csv(ruta, cabeceras, filas):  # Exporta una tabla legible en Excel sin fórmulas complejas.
    with open(ruta, 'w', encoding='utf-8-sig', newline='') as archivo:  # Escribe UTF-8 con BOM para acentos en Excel.
        escritor = csv.DictWriter(archivo, fieldnames=cabeceras)  # Fija el orden de las columnas.
        escritor.writeheader()  # Incorpora los nombres de campo en la primera fila.
        escritor.writerows(filas)  # Guarda todas las filas de datos con separadores estándar.


def convertir_fila_pso(p):  # Descompone el registro PSO vectorial en columnas numéricas.
    fila = {'iteracion': p['iteracion'], 'individuo': p['individuo'], 'fp_ant': p['fp_ant'], 'fg_ant': p['fg_ant'], 'w': p['w'], 'c1': p['c1'], 'c2': p['c2'], 'f': p['f'], 'mejora': int(p['mejora']), 'fp': p['fp']}  # Añade campos escalares.
    for clave in ('x_ant', 'v_ant', 'p_ant', 'g_ant', 'r1', 'r2', 'inercia', 'cognitivo', 'social', 'v_raw', 'v', 'x_raw', 'x', 'p'):  # Recorre vectores del paso.
        for d in range(D):  # Separa x1, x2 y x3 con el mismo orden para todos los vectores.
            fila[f'{clave}_{d+1}'] = float(p[clave][d])  # Conserva precisión doble en cada celda.
    return fila  # Devuelve una fila plana de la traza detallada.


def convertir_fila_de(p):  # Descompone cada generación DE en componentes trazables.
    fila = {clave: p[clave] for clave in ('iteracion', 'individuo', 'f_ant', 'idx_best', 'r1idx', 'r2idx', 'F', 'CR', 'jrand', 'fprueba', 'f')}  # Conserva escalares y elecciones aleatorias.
    fila['acepta'] = int(p['acepta'])  # Codifica selección como cero o uno.
    for clave in ('x_ant', 'best_ant', 'xr1', 'xr2', 'azar', 'donante_raw', 'donante', 'prueba', 'x'):  # Agrupa vectores de entrada y salida.
        for d in range(D):  # Procesa tres coordenadas.
            fila[f'{clave}_{d+1}'] = float(p[clave][d])  # Escribe el valor exacto calculado.
            if clave == 'prueba':  # Añade indicadores del cruce junto al vector de prueba.
                fila[f'cruce_{d+1}'] = int(p['mascara'][d])  # Uno implica usar donante y cero objetivo.
    return fila  # Entrega el registro completo de una mutación/recombinación/selección.


def tabla_evolucion(resultado):  # Produce la historia para Excel y para graficación.
    return [{'iteracion': e['iteracion'], 'mejor_poblacion': e['actual'], 'mejor_historico': e['historico'], 'mu1': float(e['mu'][0]), 'mu2': float(e['mu'][1]), 'mu3': float(e['mu'][2]), 'radio': e['radio'], 'mejor_x1': float(e['mejor_x'][0]), 'mejor_x2': float(e['mejor_x'][1]), 'mejor_x3': float(e['mejor_x'][2])} for e in resultado['evolucion']]  # Conserva 31 filas desde iteración cero.


def graficar_convergencia(pso, de):  # Grafica el mínimo acumulado y el mínimo de la población actual.
    t = np.arange(T + 1)  # Construye el eje horizontal desde generación 0 hasta 30.
    fig, ejes = plt.subplots(2, 1, figsize=(11.5, 8.3), sharex=True, constrained_layout=True)  # Dedica una fila a cada definición de mínimo.
    for ax, clave, titulo in zip(ejes, ('historico', 'actual'), ('(a) Mejor solución encontrada hasta el momento', '(b) Mejor solución en la población de cada iteración')):  # Recorre las dos métricas exigidas.
        ax.plot(t, [e[clave] for e in pso['evolucion']], '-o', linewidth=1.6, markersize=3.1, label='PSO–TVAC')  # Dibuja la trayectoria PSO.
        ax.plot(t, [e[clave] for e in de['evolucion']], '-s', linewidth=1.6, markersize=3.0, label='DE/current-to-best/1/bin')  # Dibuja la trayectoria DE.
        ax.set_yscale('symlog', linthresh=1e-3)  # Mantiene visibles los valores cercanos a cero y los iniciales altos.
        ax.set_title(titulo, fontweight='bold', loc='left')  # Titula el panel según el inciso solicitado.
        ax.set_ylabel('f₃(x) (escala simlog)')  # Indica la función minimizada y la escala gráfica.
        ax.grid(True, alpha=0.25)  # Facilita leer convergencias y diferencias.
        ax.legend(loc='upper right')  # Identifica ambos algoritmos sin ambigüedad.
    ejes[-1].set_xlabel('Iteración / generación')  # Rotula el eje horizontal compartido.
    fig.suptitle('Equipo 3 · Función Ackley · 30 individuos · 30 iteraciones', fontsize=14, fontweight='bold')  # Resume el experimento.
    fig.savefig(OUT / '01_convergencia.png', dpi=220, bbox_inches='tight')  # Guarda la gráfica a alta resolución.
    plt.close(fig)  # Libera la memoria de la figura terminada.


def graficar_radio(pso, de):  # Grafica ζ=max_j ||x_j−μ||₂ en cada generación.
    t = np.arange(T + 1)  # Crea el eje de iteraciones, incluida la población inicial.
    fig, ax = plt.subplots(figsize=(11.5, 4.9), constrained_layout=True)  # Crea una figura exclusiva para el radio.
    ax.plot(t, [e['radio'] for e in pso['evolucion']], '-o', markersize=3, label='PSO–TVAC')  # Añade el radio poblacional de PSO.
    ax.plot(t, [e['radio'] for e in de['evolucion']], '-s', markersize=3, label='DE/current-to-best/1/bin')  # Añade el radio poblacional de DE.
    ax.set(xlabel='Iteración / generación', ylabel='Radio poblacional ζ (norma euclidiana)', title='(d) Radio poblacional en cada iteración')  # Rotula las magnitudes.
    ax.grid(True, alpha=0.25)  # Añade líneas guía para comparar dispersión.
    ax.legend()  # Muestra las dos denominaciones de los métodos.
    fig.savefig(OUT / '02_radio_poblacional.png', dpi=220, bbox_inches='tight')  # Exporta la figura completa.
    plt.close(fig)  # Cierra la figura para liberar recursos.


def graficar_cajas(resultado, nombre, archivo):  # Dibuja 30 diagramas de caja para cada una de las tres dimensiones.
    fig, ejes = plt.subplots(D, 1, figsize=(15, 11), sharex=True, constrained_layout=True)  # Crea un panel independiente por coordenada.
    for d, ax in enumerate(ejes):  # Recorre las dimensiones x1, x2 y x3.
        muestras = [e['coordenadas'][:, d] for e in resultado['evolucion'][1:]]  # Agrupa las 30 poblaciones posteriores a cada actualización.
        ax.boxplot(muestras, positions=np.arange(1, T + 1), widths=0.55, showfliers=True, manage_ticks=False)  # Calcula mediana, cuartiles, bigotes y atípicos reales.
        ax.set_ylabel(f'Dimensión x{d + 1}')  # Identifica la dimensión que se está examinando.
        ax.grid(axis='y', alpha=0.22)  # Ayuda a comparar la dispersión entre generaciones.
        ax.axhline(0, linestyle=':', linewidth=0.9)  # Marca el mínimo teórico en la coordenada cero.
    ejes[-1].set_xticks(np.arange(1, T + 1))  # Etiqueta las 30 iteraciones representadas.
    ejes[-1].set_xlabel('Iteración / generación')  # Indica el significado del eje horizontal.
    fig.suptitle(f'(c) Cajas y bigotes · {nombre} · Tres dimensiones', fontsize=15, fontweight='bold')  # Titula la figura.
    fig.savefig(OUT / archivo, dpi=220, bbox_inches='tight')  # Genera imagen de alta calidad.
    plt.close(fig)  # Cierra la figura una vez guardada.


def main():  # Coordina ajuste de parámetros, experimento principal y exportación.
    ajuste_pso = ajustar('PSO', candidatos_pso(), pso_tvac)  # Identifica el mejor TVAC de las 12 combinaciones.
    ajuste_de = ajustar('DE', candidatos_de(), de_current_to_best)  # Identifica el mejor DE de las 12 combinaciones.
    p_pso = {c: ajuste_pso[0][c] for c in ('c10', 'c11', 'c20', 'c21', 'w0', 'w1', 'vfrac')}  # Extrae solamente los hiperparámetros PSO.
    p_de = {c: ajuste_de[0][c] for c in ('F', 'CR')}  # Extrae solamente los hiperparámetros DE.
    x0 = punto_inicial(BASE_SEED)  # Inicializa exactamente la misma población para ambos métodos.
    pso = pso_tvac(x0, p_pso, BASE_SEED + 100000, guardar=True)  # Ejecuta 30 iteraciones PSO con detalle auditable.
    de = de_current_to_best(x0, p_de, BASE_SEED + 200000, guardar=True)  # Ejecuta 30 iteraciones DE con detalle auditable.
    pso_traza = [convertir_fila_pso(fila) for fila in pso['pasos']]  # Aplana todos los pasos PSO para CSV.
    de_traza = [convertir_fila_de(fila) for fila in de['pasos']]  # Aplana todos los pasos DE para CSV.
    escribir_csv(OUT / 'ajuste_PSO.csv', list(ajuste_pso[0]), ajuste_pso)  # Guarda las 12 configuraciones PSO ordenadas.
    escribir_csv(OUT / 'ajuste_DE.csv', list(ajuste_de[0]), ajuste_de)  # Guarda las 12 configuraciones DE ordenadas.
    escribir_csv(OUT / 'PSO_pasos.csv', list(pso_traza[0]), pso_traza)  # Guarda 900 filas de actualizaciones PSO.
    escribir_csv(OUT / 'DE_pasos.csv', list(de_traza[0]), de_traza)  # Guarda 900 filas de actualizaciones DE.
    escribir_csv(OUT / 'PSO_evolucion.csv', list(tabla_evolucion(pso)[0]), tabla_evolucion(pso))  # Guarda 31 métricas por iteración PSO.
    escribir_csv(OUT / 'DE_evolucion.csv', list(tabla_evolucion(de)[0]), tabla_evolucion(de))  # Guarda 31 métricas por generación DE.
    replicas = []  # Prepara una evaluación adicional con 20 poblaciones independientes.
    for semilla in TEST_SEEDS:  # Repite la comparación fuera de las semillas de ajuste.
        inicio = punto_inicial(semilla)  # Comparte el mismo inicio en esta repetición.
        prueba_pso = pso_tvac(inicio, p_pso, semilla + 100000)  # Resuelve con los parámetros PSO ya seleccionados.
        prueba_de = de_current_to_best(inicio, p_de, semilla + 200000)  # Resuelve con los parámetros DE ya seleccionados.
        replicas.append({'semilla': semilla, 'PSO': float(prueba_pso['mejor']), 'DE': float(prueba_de['mejor']), 'PSO_gana': int(prueba_pso['mejor'] < prueba_de['mejor'])})  # Conserva ambos finales para comparación pareada.
    escribir_csv(OUT / 'comparacion_20_repeticiones.csv', list(replicas[0]), replicas)  # Exporta los 20 pares comparables.
    graficar_convergencia(pso, de)  # Crea los dos paneles solicitados en (a) y (b).
    graficar_radio(pso, de)  # Crea el panel del radio ζ solicitado en (d).
    graficar_cajas(pso, 'PSO–TVAC', '03_boxplots_PSO.png')  # Crea las cajas por dimensión de PSO para (c).
    graficar_cajas(de, 'DE/current-to-best/1/bin', '04_boxplots_DE.png')  # Crea las cajas por dimensión de DE para (c).
    informe = {'funcion': 'Ackley 3D', 'optimo_teorico': {'x': [0, 0, 0], 'f': 0}, 'dominio': [LOW, HIGH], 'poblacion': N, 'iteraciones': T, 'semilla_principal': BASE_SEED, 'semillas_calibracion': TUNE_SEEDS, 'semillas_validacion': TEST_SEEDS, 'criterio_ajuste': 'menor media f_final en 8 corridas (desempate: mediana y desviación)', 'pso_parametros': p_pso, 'de_parametros': p_de, 'pso_final': float(pso['mejor']), 'pso_x': pso['mejor_x'].tolist(), 'de_final': float(de['mejor']), 'de_x': de['mejor_x'].tolist(), 'pso_validacion_media': float(np.mean([e['PSO'] for e in replicas])), 'de_validacion_media': float(np.mean([e['DE'] for e in replicas])), 'pso_victorias': sum(e['PSO_gana'] for e in replicas), 'restricciones': 'recorte de posiciones, donantes y velocidades al dominio indicado'}  # Reúne diseño experimental y resultados.
    with open(OUT / 'resultados.json', 'w', encoding='utf-8') as archivo:  # Abre el informe legible en cualquier editor.
        json.dump(informe, archivo, indent=2, ensure_ascii=False)  # Guarda todos los números sin redondeo artificial.
    print(json.dumps(informe, indent=2, ensure_ascii=False))  # Muestra en consola el resumen matemático.


if __name__ == '__main__':  # Ejecuta el experimento solamente cuando se llama el archivo directamente.
    main()  # Lanza ajuste, comparación y exportación reproducible.