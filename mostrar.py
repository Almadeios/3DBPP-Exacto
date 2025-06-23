import trimesh
import sys

# Ruta base de tus archivos .obj
BASE_PATH = 'kitchen/objects/'

# Nombre del archivo que quieres mostrar (puedes cambiarlo)
archivo = 'gd_rubber_duck_poisson_001_scaled.obj.smoothed_5.obj'

# Carga y visualización
ruta_completa = BASE_PATH + archivo
mesh = trimesh.load(ruta_completa)

mesh.show()