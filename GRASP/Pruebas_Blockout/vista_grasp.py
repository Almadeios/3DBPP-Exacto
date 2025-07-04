# vista_ga.py
import os
import json
import trimesh
import pyrender
import numpy as np
import random

# Rutas
OBJ_DIR = "dataset/blockout/shape_vhacd"
INPUT_JSON = "GRASP/Pruebas_Blockout/solucion_grasp_blockout.json"

# Cargar objetos colocados
with open(INPUT_JSON) as f:
    placed = json.load(f)

# Crear escena de pyrender
scene = pyrender.Scene()

# Contenedor transparente
container_dims = np.array([320, 320, 300]) / 1000.0
container_box = trimesh.creation.box(extents=container_dims)
container_box.visual.face_colors = [150, 200, 255, 40]
container_tf = np.eye(4)
container_tf[:3, 3] = container_dims / 2.0  # Centrado
scene.add(pyrender.Mesh.from_trimesh(container_box, smooth=False), pose=container_tf)

# Agregar objetos
for item in placed:
    obj_path = os.path.join(OBJ_DIR, item["id"])
    mesh = trimesh.load(obj_path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()

    # Color aleatorio suave
    color = [random.randint(150, 255) for _ in range(3)] + [255]
    mesh.visual.face_colors = np.tile(color, (len(mesh.faces), 1))

    # Posición aplicada directamente
    tf = np.eye(4)
    tf[:3, 3] = item["position_m"]
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False), pose=tf)

# Calcular volumen total del contenedor
volumen_total = np.prod(container_dims)

# Calcular volumen ocupado por los objetos colocados
volumen_ocupado = 0.0
for item in placed:
    mesh = trimesh.load(os.path.join(OBJ_DIR, item["id"]), force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()
    volumen_ocupado += mesh.volume

#resultados
porcentaje_ocupado = (volumen_ocupado / volumen_total) * 100
print(f"\n Objetos colocados: {len(placed)}")
print(f"Volumen contenedor: {volumen_total:.6f} m³")
print(f"Volumen usado:     {volumen_ocupado:.6f} m³")
print(f"Porcentaje lleno:  {porcentaje_ocupado:.2f}%")


# Mostrar
pyrender.Viewer(scene, use_raymond_lighting=True, point_size=2)


