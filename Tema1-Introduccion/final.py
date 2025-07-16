# Simulación completa de un autómata celular 3D para anodización con animación 3D y estadísticas
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import time
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation

# --- Definición de estados ---
M = 0   # Metal
OX = 1  # Óxido
EF = 2  # Campo eléctrico (Electric Field walker)
A = 3   # Aniones
S = 4   # Solvente

state_colors = {
    M: 'gray',
    OX: 'red',
    EF: 'blue',
    A: 'green',
    S: 'white'
}

# --- Configuración de la rejilla ---
Nx, Ny, Nz = 30, 30, 30  # Tamaño reducido para visualización

grid = np.full((Nx, Ny, Nz), S, dtype=np.int8)  # Iniciar con solvente

# Asignar metal en la parte inferior
grid[:, :, :5] = M

# --- Vecindad de Moore en 3D ---
neighbors = [offset for offset in product([-1, 0, 1], repeat=3) if offset != (0, 0, 0)]

# --- Parámetro para probabilidad de ruptura de enlace ---
PBOND = 0.9

# --- Registro de estadísticas ---
state_counts = []

# --- Reglas del modelo completo según el artículo ---
def rule_1_passivation(i, j, k, ni, nj, nk):
    if grid[i,j,k] == M and grid[ni,nj,nk] == S:
        grid[i,j,k] = OX
        grid[ni,nj,nk] = OX

def rule_2_field_dissolution(i, j, k, ni, nj, nk):
    if grid[i,j,k] == EF and grid[ni,nj,nk] == S:
        grid[ni,nj,nk] = S

def rule_3_field_to_anion(i, j, k, ni, nj, nk):
    if grid[i,j,k] == EF and grid[ni,nj,nk] == S:
        grid[ni,nj,nk] = A

def rule_4_oxidation_by_anion(i, j, k, ni, nj, nk):
    if grid[i,j,k] == M and grid[ni,nj,nk] == A:
        grid[i,j,k] = OX
        grid[ni,nj,nk] = OX

def rule_5_create_field(i, j, k, ni, nj, nk):
    if grid[i,j,k] == M and grid[ni,nj,nk] == OX:
        grid[ni,nj,nk] = EF

def rule_6_diffuse_field(i, j, k, ni, nj, nk):
    if grid[i,j,k] == EF and grid[ni,nj,nk] == OX:
        grid[i,j,k] = OX
        grid[ni,nj,nk] = EF

def rule_7_diffuse_anion(i, j, k, ni, nj, nk):
    if grid[i,j,k] == EF and grid[ni,nj,nk] == A:
        grid[i,j,k] = A
        grid[ni,nj,nk] = EF

def rule_8_surface_reorg(i, j, k, ni, nj, nk):
    if grid[i,j,k] == S and grid[ni,nj,nk] == OX:
        grid[i,j,k] = OX
        grid[ni,nj,nk] = S

def rule_9_surface_reorg_anion(i, j, k, ni, nj, nk):
    if grid[i,j,k] == S and grid[ni,nj,nk] == A:
        grid[i,j,k] = A
        grid[ni,nj,nk] = S

# --- Aplicar todas las reglas ---
def apply_rules():
    for i in range(1, Nx-1):
        for j in range(1, Ny-1):
            for k in range(1, Nz-1):
                for dx, dy, dz in neighbors:
                    ni, nj, nk = i+dx, j+dy, k+dz
                    rule_1_passivation(i,j,k,ni,nj,nk)
                    rule_2_field_dissolution(i,j,k,ni,nj,nk)
                    rule_3_field_to_anion(i,j,k,ni,nj,nk)
                    rule_4_oxidation_by_anion(i,j,k,ni,nj,nk)
                    rule_5_create_field(i,j,k,ni,nj,nk)
                    rule_6_diffuse_field(i,j,k,ni,nj,nk)
                    rule_7_diffuse_anion(i,j,k,ni,nj,nk)
                    rule_8_surface_reorg(i,j,k,ni,nj,nk)
                    rule_9_surface_reorg_anion(i,j,k,ni,nj,nk)

# --- Contar estados ---
def count_states():
    counts = [np.sum(grid == s) for s in range(5)]
    state_counts.append(counts)

# --- Crear animación 3D ---
def animate_simulation(steps):
    fig = plt.figure(figsize=(8,6))
    ax = fig.add_subplot(111, projection='3d')

    def update(frame):
        apply_rules()
        count_states()
        ax.clear()
        ax.set_xlim(0, Nx)
        ax.set_ylim(0, Ny)
        ax.set_zlim(0, Nz)
        ax.set_title(f"Paso {frame}")

        for state, color in state_colors.items():
            x, y, z = np.where(grid == state)
            ax.scatter(x, y, z, c=color, label=f"{state}", s=2)

    anim = animation.FuncAnimation(fig, update, frames=steps, interval=300, repeat=False)
    plt.show()

    # --- Graficar estadísticas después de la animación ---
    labels = ['Metal', 'Óxido', 'Campo', 'Aniones', 'Solvente']
    data = np.array(state_counts)
    plt.figure(figsize=(10,5))
    for i in range(5):
        plt.plot(data[:,i], label=labels[i])
    plt.xlabel("Paso de tiempo")
    plt.ylabel("Cantidad de celdas")
    plt.title("Evolución de los estados durante la anodización")
    plt.legend()
    plt.grid(True)
    plt.show()

# --- Ejecutar animación ---
animate_simulation(30)