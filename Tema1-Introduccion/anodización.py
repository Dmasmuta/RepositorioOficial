import numpy as np
from numba import cuda 
from numba import jit
from numba import prange
from numba import float32, int32
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.animation import FuncAnimation
import time
import math
import matplotlib

# Configurar backend para renderizado más rápido
matplotlib.use('TkAgg')  # Usar 'Qt5Agg' si tienes PyQt/PySide instalado

# ========================
# CONSTANTES Y CONFIGURACIÓN
# ========================
# Estados celulares
M = 0    # Metal
OX = 1   # Óxido
EF = 2   # Campo eléctrico
A = 3    # Anión
S = 4    # Solvente

# Parámetros del modelo (valores típicos del artículo)
P_DISSOLUTION = 0.15  # Prob. disolución óxido (regla 6)
P_ANION = 0.5         # Prob. incorporación aniones (regla 7)
P_EF_GEN = 0.8        # Prob. generación EF (regla 9)
P_DIFF = 1.0          # Prob. difusión (reglas 10,11)
P_BOND = 0.1          # Prob. "desenlace" (reglas 12,13)

# Tamaño de la red (ajustar según GPU)
NX, NY, NZ = 50, 50, 100  # Tamaño reducido para mejor rendimiento en animación

# Tamaño de bloque CUDA
THREADS_PER_BLOCK = 8
BLOCKS_X = math.ceil(NX / THREADS_PER_BLOCK)
BLOCKS_Y = math.ceil(NY / THREADS_PER_BLOCK)
BLOCKS_Z = math.ceil(NZ / THREADS_PER_BLOCK)

# Vecindad de Moore (26 vecinos en 3D)
moore_neighborhood = np.array([(dx, dy, dz) 
                              for dx in (-1, 0, 1) 
                              for dy in (-1, 0, 1) 
                              for dz in (-1, 0, 1) 
                              if (dx, dy, dz) != (0, 0, 0)], dtype=np.int32)

# ========================
# INICIALIZACIÓN EN GPU
# ========================
@cuda.jit
def init_grid_kernel(grid, metal_thickness, seed_z):
    z, y, x = cuda.grid(3)
    
    if x < NX and y < NY and z < NZ:
        # Capa metálica inferior
        if z < metal_thickness:
            grid[z, y, x] = M
        # Semillas de óxido en zona de inicio
        elif z < seed_z and z % 3 == 0 and x % 3 == 0 and y % 3 == 0:
            grid[z, y, x] = OX
        else:
            grid[z, y, x] = S

def initialize_grid_3d(metal_thickness=5, seed_z=10):
    grid = cuda.device_array((NZ, NY, NX), dtype=np.uint8)
    
    # Configurar kernel
    threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK)
    blockspergrid = (BLOCKS_X, BLOCKS_Y, BLOCKS_Z)
    
    init_grid_kernel[blockspergrid, threadsperblock](
        grid, metal_thickness, seed_z
    )
    
    return grid

# ========================
# KERNELS DE SIMULACIÓN
# ========================
@cuda.jit
def reaction_kernel(grid, new_grid, params, step, rand_seed):
    z, y, x = cuda.grid(3)
    
    if x >= NX or y >= NY or z >= NZ or z == 0 or z == NZ-1:
        return
    
    # Estados actuales
    site = grid[z, y, x]
    
    # Semilla aleatoria basada en posición y paso
    seed = x * 137 + y * 149 + z * 101 + step + rand_seed
    random_val = ((seed * 1103515245) + 12345) & 0x7fffffff
    rand = (random_val % 10000) / 10000.0
    
    # Seleccionar vecino aleatorio
    neighbor_idx = random_val % 26
    dz, dy, dx = moore_neighborhood[neighbor_idx]
    nz, ny, nx = z + dz, y + dy, x + dx
    
    # Condiciones de contorno
    if nz < 0 or nz >= NZ: 
        return
    nx = nx % NX  # Periódico en X
    ny = ny % NY  # Periódico en Y
    
    neighbor = grid[nz, ny, nx]
    
    # 1. REGLAS DE REACCIÓN
    # Pasivación del metal (regla 5): M + S -> OX + OX
    if (site == M and neighbor == S) or (site == S and neighbor == M):
        new_grid[z, y, x] = OX
        new_grid[nz, ny, nx] = OX
        return
    
    # Disolución del óxido (regla 6): EF + S -> S + S
    elif site == EF and neighbor == S and rand < params[0]:
        new_grid[z, y, x] = S
        return
    
    # Incorporación de aniones (regla 7): EF + S -> A + S
    elif site == EF and neighbor == S and rand < params[1]:
        new_grid[z, y, x] = A
        return
    
    # Generación de EF (regla 9): M + OX -> M + EF
    elif site == M and neighbor == OX and rand < params[2]:
        new_grid[nz, ny, nx] = EF
        return
    
    # Oxidación por aniones (regla 8): M + A -> OX + OX
    elif site == M and neighbor == A:
        new_grid[z, y, x] = OX
        new_grid[nz, ny, nx] = OX
        return
    
    # 2. REGLAS DE DIFUSIÓN
    # Difusión EF-OX (regla 10)
    elif ((site == EF and neighbor == OX) or 
          (site == OX and neighbor == EF)) and rand < params[3]:
        new_grid[z, y, x] = neighbor
        new_grid[nz, ny, nx] = site
        return
    
    # Difusión EF-A (regla 11)
    elif ((site == EF and neighbor == A) or 
          (site == A and neighbor == EF)) and rand < params[3]:
        new_grid[z, y, x] = neighbor
        new_grid[nz, ny, nx] = site
        return
    
    # 3. REORGANIZACIÓN SUPERFICIAL
    # Intercambio OX-S (regla 12) y A-S (regla 13)
    elif ((site == OX and neighbor == S) or 
          (site == A and neighbor == S) or
          (site == S and neighbor == OX) or 
          (site == S and neighbor == A)):
        
        # Contar vecinos tipo-óxido
        count_site = 0
        count_neighbor = 0
        for i in range(26):
            dz2, dy2, dx2 = moore_neighborhood[i]
            z2, y2, x2 = z + dz2, (y + dy2) % NY, (x + dx2) % NX
            if 0 <= z2 < NZ:
                if grid[z2, y2, x2] in (OX, EF, A):
                    count_site += 1
            
            z2, y2, x2 = nz + dz2, (ny + dy2) % NY, (nx + dx2) % NX
            if 0 <= z2 < NZ:
                if grid[z2, y2, x2] in (OX, EF, A):
                    count_neighbor += 1
        
        # Aplicar reglas de intercambio
        if ((site == S and count_site > count_neighbor) or
            (neighbor == S and count_neighbor > count_site)):
            new_grid[z, y, x] = neighbor
            new_grid[nz, ny, nx] = site
        else:
            N_diff = abs(count_site - count_neighbor)
            P_swap = params[4] ** N_diff
            if rand < P_swap:
                new_grid[z, y, x] = neighbor
                new_grid[nz, ny, nx] = site

# ========================
# CLASE DE SIMULACIÓN
# ========================
class AnodizationSimulation:
    def __init__(self, steps=1000):
        # Configurar parámetros
        self.params = np.array([
            P_DISSOLUTION,  # 0
            P_ANION,        # 1
            P_EF_GEN,       # 2
            P_DIFF,         # 3
            P_BOND          # 4
        ], dtype=np.float32)
        
        # Inicializar grid en GPU
        self.grid = initialize_grid_3d(metal_thickness=5, seed_z=10)
        self.new_grid = cuda.device_array((NZ, NY, NX), dtype=np.uint8)
        
        # Configurar kernel
        self.threadsperblock = (THREADS_PER_BLOCK, THREADS_PER_BLOCK, THREADS_PER_BLOCK)
        self.blockspergrid = (BLOCKS_X, BLOCKS_Y, BLOCKS_Z)
        
        self.step = 0
        self.total_steps = steps
        self.rand_seed = int(time.time() * 1000)
        
        # Configurar figura para animación
        self.fig, self.axes = plt.subplots(1, 2, figsize=(16, 8))
        self.fig.suptitle("Simulación 3D de Anodización - Autómata Celular", fontsize=16)
        
        # Paleta de colores
        self.cmap = colors.ListedColormap(['gray', 'black', 'yellow', 'red', 'blue'])
        self.bounds = [0, 1, 2, 3, 4, 5]
        self.norm = colors.BoundaryNorm(self.bounds, self.cmap.N)
        
        # Cortes iniciales
        self.grid_cpu = self.grid.copy_to_host()
        
        # Corte XY (plano horizontal)
        self.im_xy = self.axes[0].imshow(
            self.grid_cpu[NZ//2], 
            cmap=self.cmap, 
            norm=self.norm,
            interpolation='nearest'
        )
        self.axes[0].set_title(f"Corte XY en Z={NZ//2}")
        self.axes[0].set_xlabel("X")
        self.axes[0].set_ylabel("Y")
        
        # Corte XZ (plano vertical)
        self.im_xz = self.axes[1].imshow(
            self.grid_cpu[:, NY//2, :], 
            cmap=self.cmap, 
            norm=self.norm,
            aspect='auto',
            interpolation='nearest'
        )
        self.axes[1].set_title(f"Corte XZ en Y={NY//2}")
        self.axes[1].set_xlabel("X")
        self.axes[1].set_ylabel("Z")
        
        # Barra de color
        cbar = self.fig.colorbar(self.im_xy, ax=self.axes, shrink=0.6)
        cbar.set_ticks([0.5, 1.5, 2.5, 3.5, 4.5])
        cbar.set_ticklabels(['Metal', 'Óxido', 'Campo Eléctrico', 'Anión', 'Solvente'])
        
        # Información de tiempo
        self.time_text = self.fig.text(0.15, 0.95, '', fontsize=12, 
                                       bbox=dict(facecolor='white', alpha=0.7))
        
        # Estadísticas
        self.ox_stats = []
        self.time_stats = []
        self.start_time = time.time()
    
    def update_simulation(self):
        if self.step >= self.total_steps:
            return False
            
        # Copiar el estado actual
        self.new_grid.copy_to_device(self.grid)
        
        # Ejecutar kernel de reacción
        reaction_kernel[self.blockspergrid, self.threadsperblock](
            self.grid, self.new_grid, self.params, self.step, self.rand_seed
        )
        
        # Intercambiar grids para el siguiente paso
        self.grid, self.new_grid = self.new_grid, self.grid
        
        # Copiar datos a CPU para visualización (cada 5 pasos)
        if self.step % 5 == 0:
            self.grid_cpu = self.grid.copy_to_host()
        
        # Actualizar estadísticas
        if self.step % 10 == 0:
            current_time = time.time() - self.start_time
            ox_count = np.sum(self.grid_cpu == OX)
            self.ox_stats.append(ox_count)
            self.time_stats.append(current_time)
        
        self.step += 1
        return True
    
    def update_plot(self, frame):
        if not self.update_simulation():
            return [self.im_xy, self.im_xz, self.time_text]
        
        # Actualizar imágenes
        self.im_xy.set_data(self.grid_cpu[NZ//2])
        self.im_xz.set_data(self.grid_cpu[:, NY//2, :])
        
        # Actualizar texto
        elapsed = time.time() - self.start_time
        fps = self.step / elapsed if elapsed > 0 else 0
        time_str = (f"Paso: {self.step}/{self.total_steps} | "
                   f"Tiempo: {elapsed:.1f}s | FPS: {fps:.1f}\n"
                   f"Óxido: {np.sum(self.grid_cpu == OX):,} celdas")
        self.time_text.set_text(time_str)
        
        return [self.im_xy, self.im_xz, self.time_text]
    
    def run_animation(self):
        # Crear animación
        self.animation = FuncAnimation(
            self.fig, 
            self.update_plot, 
            frames=self.total_steps,
            interval=50,  # ms entre frames
            blit=True
        )
        
        plt.tight_layout(rect=[0, 0, 1, 0.95])  # Ajustar para el texto superior
        plt.show()
    
    def save_animation(self, filename):
        # Guardar animación como video
        from matplotlib.animation import FFMpegWriter
        writer = FFMpegWriter(fps=15, bitrate=1800)
        self.animation.save(filename, writer=writer)

# ========================
# EJECUCIÓN PRINCIPAL
# ========================
if __name__ == "__main__":
    # Crear y ejecutar simulación con animación
    sim = AnodizationSimulation(steps=500)
    sim.run_animation()