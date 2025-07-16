import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.animation as animation
from IPython.display import HTML
import random

# ====================
# PARÁMETROS DEL MODELO
# ====================
NX, NY = 50, 50    # Tamaño de la red
STEPS = 500         # Número de pasos de simulación

# Estados (siguiendo el artículo)
M = 0    # Metal
OX = 1   # Óxido
EF = 2   # Campo eléctrico
A = 3    # Anión
S = 4    # Solvente

# Probabilidades (valores típicos del artículo)
P_DISSOLUTION = 0.15  # Prob. disolución óxido
P_BOND = 0.1          # Prob. "desenlace"
P_ANION = 0.5         # Prob. incorporación aniones

# ====================
# INICIALIZACIÓN
# ====================
def initialize_grid():
    grid = np.full((NY, NX), S, dtype=int)
    
    # Capa inferior de metal
    grid[0:5, :] = M
    
    # Semilla para poros (opcional)
    for _ in range(20):
        x, y = random.randint(0, NX-1), random.randint(5, 10)
        grid[y, x] = OX
        
    return grid

# ====================
# REGLAS DEL AUTÓMATA
# ====================
def apply_rules(grid):
    new_grid = grid.copy()
    
    for _ in range(NX * NY):  # Actualizar todos los sitios
        # Seleccionar sitio aleatorio
        i, j = random.randint(1, NY-2), random.randint(0, NX-1)
        
        # Seleccionar vecino aleatorio (Moore neighborhood)
        di, dj = random.choice([(0,1), (1,0), (0,-1), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)])
        ni, nj = i + di, (j + dj) % NX  # Periodicidad en X
        
        # Evitar bordes superior/inferior
        if ni < 0 or ni >= NY: 
            continue
            
        # Estados actuales
        current = grid[i, j]
        neighbor = grid[ni, nj]
        
        # 1. REGLAS DE REACCIÓN (Artículo: ecs. 5-9)
        if (current == M and neighbor == S) or (current == S and neighbor == M):
            new_grid[i, j] = OX
            new_grid[ni, nj] = OX  # Pasivación del metal
            
        elif current == EF and neighbor == S:
            if random.random() < P_DISSOLUTION:
                new_grid[i, j] = S  # Disolución del óxido
            elif random.random() < P_ANION:
                new_grid[ni, nj] = A  # Incorporación de aniones
                
        elif current == M and neighbor == OX:
            new_grid[ni, nj] = EF  # Generación de campo eléctrico
            
        # 2. REGLAS DE DIFUSIÓN (Artículo: ecs. 10-11)
        elif (current == EF and neighbor == OX) or (current == OX and neighbor == EF):
            # Intercambio EF-OX (difusión)
            new_grid[i, j], new_grid[ni, nj] = neighbor, current
            
        # 3. REORGANIZACIÓN SUPERFICIAL (Artículo: ecs. 12-13)
        elif (current == S and neighbor == OX) or (current == OX and neighbor == S):
            if random.random() < P_BOND:
                new_grid[i, j], new_grid[ni, nj] = neighbor, current
    
    # Mantener capa metálica inferior
    new_grid[0:2, :] = M
    return new_grid

# ====================
# SIMULACIÓN Y VISUALIZACIÓN
# ====================
def run_simulation():
    grid = initialize_grid()
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Paleta de colores (basada en estados)
    cmap = colors.ListedColormap(['gray', 'black', 'yellow', 'red', 'blue'])
    bounds = [0, 1, 2, 3, 4, 5]
    norm = colors.BoundaryNorm(bounds, cmap.N)
    
    img = ax.imshow(grid, cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_title("Simulación de Anodización por Autómatas Celulares")
    
    def update(frame):
        nonlocal grid
        grid = apply_rules(grid)
        img.set_data(grid)
        return [img]
    
    ani = animation.FuncAnimation(fig, update, frames=STEPS, 
                                  interval=50, blit=True)
    
    plt.close()
    return HTML(ani.to_jshtml())

# Ejecutar la simulación
run_simulation()
