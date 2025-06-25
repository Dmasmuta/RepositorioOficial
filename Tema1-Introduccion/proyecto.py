import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors

# Configuración
tamaño = 30  # Tamaño de la cuadrícula (30x30)
pasos = 50    # Número de iteraciones
p = 0.7       # Prob. de ignición con vecino ardiente
f = 0.01      # Prob. de crecimiento de nuevo árbol

# Estados: 0=VACÍO, 1=ÁRBOL, 2=LLAMAS, 3=CENIZA
grid = np.zeros((tamaño, tamaño))

# Inicializar: 50% árboles, un foco inicial de fuego
grid = np.random.choice([0, 1], size=(tamaño, tamaño), p=[0.5, 0.5])
grid[15, 15] = 2  # Foco inicial de incendio

# Colores para visualizar
cmap = colors.ListedColormap(['white', 'green', 'red', 'black'])
bounds = [0, 1, 2, 3, 4]
norm = colors.BoundaryNorm(bounds, cmap.N)

for _ in range(pasos):
    nuevo_grid = grid.copy()
    for i in range(tamaño):
        for j in range(tamaño):
            # Regla 1: Árbol en llamas -> Ceniza
            if grid[i, j] == 2:
                nuevo_grid[i, j] = 3
            # Regla 2: Árbol sano puede incendiarse
            elif grid[i, j] == 1:
                vecinos = grid[max(i-1,0):min(i+2,tamaño), max(j-1,0):min(j+2,tamaño)]
                if 2 in vecinos and np.random.random() < p:
                    nuevo_grid[i, j] = 2
            # Regla 3: Regeneración en celdas vacías
            elif grid[i, j] == 0 and np.random.random() < f:
                nuevo_grid[i, j] = 1
    grid = nuevo_grid.copy()
    
    # Visualizar (requiere matplotlib)
    plt.imshow(grid, cmap=cmap, norm=norm)
    plt.pause(0.1)
plt.show()