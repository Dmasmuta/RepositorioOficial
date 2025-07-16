# Simulación básica de un autómata celular 3D para anodización en Python
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import time

# --- Definición de estados ---
M = 0   # Metal
OX = 1  # Óxido
EF = 2  # Campo eléctrico
A = 3   # Aniones
S = 4   # Solvente

# --- Configuración de la rejilla ---
Nx, Ny, Nz = 50, 50, 100  # Tamaño reducido para pruebas

grid = np.full((Nx, Ny, Nz), S, dtype=np.int8)  # Iniciar con solvente

# Asignar metal en la parte inferior (por ejemplo, primeros 10 planos en Z)
grid[:, :, :10] = M

# --- Vecindad de Moore en 3D ---
neighbors = [offset for offset in product([-1, 0, 1], repeat=3) if offset != (0, 0, 0)]

# --- Reglas simples ---
def rule_passivation(i, j, k, ni, nj, nk):
    if grid[i,j,k] == M and grid[ni,nj,nk] == S:
        grid[i,j,k] = OX
        grid[ni,nj,nk] = OX

def rule_oxidation_by_anion(i, j, k, ni, nj, nk):
    if grid[i,j,k] == M and grid[ni,nj,nk] == A:
        grid[i,j,k] = OX
        grid[ni,nj,nk] = OX

def rule_create_field(i, j, k, ni, nj, nk):
    if grid[i,j,k] == M and grid[ni,nj,nk] == OX:
        grid[ni,nj,nk] = EF

# --- Aplicar reglas sobre todos los sitios ---
def apply_rules():
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            for k in range(1, Nz-1):
                for dx, dy, dz in neighbors:
                    ni, nj, nk = i+dx, j+dy, k+dz
                    rule_passivation(i,j,k,ni,nj,nk)
                    rule_oxidation_by_anion(i,j,k,ni,nj,nk)
                    rule_create_field(i,j,k,ni,nj,nk)

# --- Simulación temporal con medición de tiempo ---
def simulate(steps):
    start = time.time()
    for t in range(steps):
        step_start = time.time()
        apply_rules()
        step_end = time.time()
        if t % 10 == 0:
            print(f"Paso {t} completado en {step_end - step_start:.2f} segundos")
    end = time.time()
    print(f"\nSimulación completa en {end - start:.2f} segundos")

# --- Visualización 2D ---
def plot_slice(z):
    plt.figure(figsize=(6,6))
    plt.imshow(grid[:,:,z], cmap='tab10', origin='lower')
    plt.title(f"Corte Z = {z}")
    plt.colorbar()
    plt.show()

# --- Ejecutar simulación ---
simulate(50)
plot_slice(20)
plot_slice(50)
plot_slice(80)
