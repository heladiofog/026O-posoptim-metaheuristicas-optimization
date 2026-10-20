# Metaheuristics Optimization Project

Este proyecto está diseñado para el desarrollo y experimentación de algoritmos de optimización metaheurística y Data Science (comenzando con **Particle Swarm Optimization (PSO)** y **Differential Evolution**). El entorno está optimizado para ser completamente multiplataforma, soportando macOS, Linux y Windows.

---

## 🛠️ Requisitos Previos

Antes de comenzar, asegúrate de tener instalado **Python 3.13 o superior** en tu sistema.
* **macOS / Linux:** Se recomienda instalar y gestionar las versiones de Python utilizando [`pyenv`](https://github.com) (recomendado).
* **Windows:** Se recomienda instalar Python desde el instalador oficial de [python.org](https://python.org) (asegurándote de marcar la casilla *"Add Python to PATH"*).

---

## 🚀 Configuración del Entorno (Instalación)

Sigue los pasos correspondientes a tu sistema operativo para clonar el proyecto y levantar el entorno virtual aislado.

### 1. Clonar el repositorio
```bash
git clone https://github.com/heladiofog/026O-posoptim-metaheuristicas-optimization.git

cd 026O-metaheuristicas-optimization
```

### 2. Crear y Activar el Entorno Virtual

#### En macOS y Linux:
```bash
# Crear el entorno virtual
python -m venv .venv

# Activar el entorno
source .venv/bin/activate
```

#### En Windows (PowerShell):
```bash
# Crear el entorno virtual
python -m venv .venv

# Activar el entorno
.venv\Scripts\Activate.ps1
```
*💡 **Nota para Windows:** Si PowerShell muestra un error de seguridad que impide la ejecución del script, abre una terminal de PowerShell como Administrador y ejecuta el comando `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser`, luego reintenta activar el entorno.*

#### En Windows (Git Bash / CMD tradicional):
```bash
# Git Bash:
source .venv/Scripts/activate

# CMD:
.venv\Scripts\activate.bat
```

---

### 📦 3. Instalar Dependencias

Una vez que veas el indicador `(.venv)` al inicio de tu terminal, actualiza las herramientas de empaquetado e instala el *stack* de librerías científicas ejecutando:

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

---

## 🛑 Desactivación del Entorno

Cuando termines de trabajar en el proyecto y quieras regresar al entorno global de tu sistema, simplemente ejecuta:
```bash
deactivate
```

---

## 📁 Estructura del Proyecto Recomendada
* `src/`: Código fuente de los algoritmos (ej. `pso.py`, `differential_evolution.py`).
* `notebooks/`: Espacio para experimentación visual y análisis de convergencia.
* `requirements.txt`: Registro estricto de librerías instaladas.

