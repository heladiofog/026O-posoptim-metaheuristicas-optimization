import numpy as np

N = 30
D = 3

semilla = 314159 #se queda fiija para que sea reproducible

rng = np.random.default_rng(semilla) #la forma en que se generan numeros aleatorios

# pestaña PSO inicio

x0 = rng.uniform(-10, 10, size=(N, D)) #el profe usaba nunmeros uniformes y aleatorios tambien puede existir bitgen me ha corrido rapido tamien bit_gen = np.random.SFC64(seed=semilla)

#print(x0)

rng_pso = np.random.default_rng(semilla+100000) #para que no salgan aleatorios repetidos a la primer lista de aleatorios

v0 = rng_pso.uniform(-6, 6, size=(N, D)) #aleatorios para velocidad #el 6 se puede ajustar para el valor que deseeemos de la velmax pestaña 

#print(v0)

for t in range(1, 31):

    r1 = rng_pso.random(size=(N,D))
    r2 = rng_pso.random(size=(N,D))

#print(r1)
#print(r2)