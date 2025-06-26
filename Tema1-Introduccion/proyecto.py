import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, patches
import matplotlib.animation as animation

# Definición de estados
NORMAL = 0     # Árbol normal (verde)
INCENDIADO = 1 # Árbol en llamas (rojo)
QUEMADO = 2    # Árbol quemado (negro)
HUMEDO = 3     # Árbol húmedo (azul)

# Tamaño del bosque (40x40 celdas)
FILAS, COLS = 40, 40

# Tamaño de cada celda en píxeles
TAM_CELDA = 20

# Probabilidades debido a agentes externos
PROB_incendio = 0.9
PROB_HUMEDAD = 0.10
PROB_SECADO = 0.10

# Variables globales para control
running = False  # Estado de la simulación

def crear_bosque():
    """Inicializa el bosque con árboles normales y algunos húmedos"""
    bosque = np.zeros((FILAS, COLS), dtype=int)
    
    # Distribución inicial de árboles húmedos
    # "fil" variable que represente el indice de las filas y "col" de la columnas
    for fil in range(FILAS):
        for col in range(COLS):
            #Los puntos de árboles húmedos con random random() es una función que devuelve un número flotante aleatorio en el rango [0.0, 1.0)
            if np.random.random() < PROB_HUMEDAD:
                bosque[fil, col] = HUMEDO  # posicicón de Arboles húmedos
            else:
                bosque[fil, col] = NORMAL
                
    return bosque

def actualizar_bosque(bosque):
    """Aplica las reglas del autómata celular"""
    if not running:
        return bosque  # Pausa la simulación
        
    nuevo_bosque = bosque.copy()
    #  comenzamos evaluar los estados de las celdas 
    for fil in range(FILAS):
        for col in range(COLS):
            estado_actual = bosque[fil, col]
            
            # 1 condición. Árbol INCENDIADO se convierte en QUEMADO
            if estado_actual == INCENDIADO:
                nuevo_bosque[fil, col] = QUEMADO
            
            # 2 condición. Árbol NORMAL que puede incendiarse
            elif estado_actual == NORMAL:
                vecinos_incendiados = 0 # Creamos un contador para los vecinos
                for desf in [-1, 0, 1]:  # Cambio de fila y columna
                    for desc in [-1, 0, 1]:
                        if desf == 0 and desc == 0: # Omitimos la celda que estamos procesando
                            continue
                        # Calculamos la posisición del vecino recorriendo las filas y columnas de alado
                        nfil, ncol = fil + desf, col + desc
                        if 0 <= nfil < FILAS and 0 <= ncol < COLS:
                            if bosque[nfil, ncol] == INCENDIADO:
                                vecinos_incendiados += 1
                
                if vecinos_incendiados > 0 and np.random.random() < PROB_incendio:
                    nuevo_bosque[fil, col] = INCENDIADO
            
            # 3 condición. Árbol HUMEDO puede secarse
            elif estado_actual == HUMEDO:
                if np.random.random() < PROB_SECADO:
                    nuevo_bosque[fil, col] = NORMAL
    
    return nuevo_bosque

# Configuración de la visualización
colomap = colors.ListedColormap(['green', 'red', 'black', 'blue']) # mapas de colores a usar
norm = colors.BoundaryNorm([0, 1, 2, 3, 4], colomap.N) # relación de colores  con numero y cantidad de colores 
etiquetas = ['Normal', 'Incendiado', 'Quemado', 'Húmedo'] # Relación  de colores con su etiqueta

# CÁLCULO DEL TAMAÑO DE LA FIGURA
DPI = 150 #Resolución porpugada
#tamañode la celda teniendo en cuenta tamañodelacelda
ancho_pulgadas = COLS * TAM_CELDA / DPI
alto_pulgadas = FILAS * TAM_CELDA / DPI

# Crear figura,en el tamaño calculado y el eje
fig, ax = plt.subplots(figsize=(ancho_pulgadas, alto_pulgadas))
bosque = crear_bosque()  # genera lamatriz de datos iniciales

# Mostrar la imagen con interpolación 'nearest' para mantener los píxeles nítidos
# imshow : que conviete la matriz en imagen coloreada,interpolation='nearest': Evita el suavizado, manteniendo bordes nítidos 
# con configuración de colores definidos en colomapy norm
imnear = ax.imshow(bosque, cmap=colomap, norm=norm, animated=True, interpolation='nearest')

# Usar Rectángulos para bordes visibles
for i in range(FILAS):
    for j in range(COLS): #despuesde recorer la matriz utilizamos patches.recta para crearuna cuadriculavisible
        recta = patches.Rectangle((j - 0.5, i - 0.5), #esquinainferiror izquierda
                                   1, #ancho
                                   1, #alto
                                linewidth=1, #Grosor delinea 
                                edgecolor='black', #color de borde 
                               facecolor='none') #rellenotransparente
        ax.add_patch(recta)

# Ocultar los ticks principales elimina las los numeros y marcas de los ejes 
ax.set_xticks([])
ax.set_yticks([])
# Texto descriptivo de la simulación
ax.set_title(" Simulación de Incendio Forestal ")

# Función para manejar clics del ratón

def comclick(event): # objeto que contiene la información del clic
    if event.inaxes == ax and event.button == 1:  # parte y si se uso el botón del Clic izquierdo
        fil, col = int(event.xdata), int(event.ydata) #coordenadas del clic
        if 0 <= col < COLS and 0 <= fil < FILAS: # verifica que sea dentro de la matriz
            bosque[col, fil ] = INCENDIADO  # actualiza el estado en la matriz
            imnear.set_array(bosque)   #actualiza el obejto de imagen  en la visualización
            fig.canvas.draw_idle()     # en lafigura se progrma un redibujo en el siclo de eventos

# Función para manejar teclado
def llaveenc(event):    # Define la función para tecla
    global running      # Accedemos a la variable global 'running' para modificarla
    if event.key == ' ':  #propiedadcon elcaracter delateclapresionada
        running = not running  # Alternar estado con barra espaciadora debido notoperador booleano invertidor
        
# Conectar eventos
fig.canvas.mpl_connect('button_press_event', comclick) #accede al liefigura, vincula event ,el tipo de event , funcion manejadora  
fig.canvas.mpl_connect('key_press_event', llaveenc) # evento  de tecla y función maneja

# Función de animación
def Animación(frame): # numero de fotogramas actual def por funcanimatión
    global bosque     # modifica la matriz
    bosque = actualizar_bosque(bosque) # llama a lacondiciones para actualizar 
    imnear.set_array(bosque)     # Actualiza la imagen de la matriz
    return imnear,   #Elemet actualizar para la tupla

# Crear animación, usamos la clase con la que temos la figura del objeto(fig), Animación función de actualización
# frame numero de fotogramas, interval tiempo entre fotogramas (ms), blit redibuja laspartes cambiadas 
Anima = animation.FuncAnimation(fig, Animación, frames=100, interval=500, blit=True)

# Ajustar márgenes para eliminar espacios en blanco
plt.tight_layout(pad=0) #Ajustar los espacios entre elementos tigt(función que evita solapamientos), parametro espacioadicional
#Ajusta manualmente la psosición y el tamaño en lascuatro direcciones
plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
#Muestra laventana con la figura Show ejecuta el resultado final de los ajustes
plt.show()
