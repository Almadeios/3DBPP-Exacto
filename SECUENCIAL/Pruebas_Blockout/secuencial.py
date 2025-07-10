# secuencial.py
import os
import json
import trimesh
import numpy as np
import torch
from trimesh.collision import CollisionManager

# Rutas
BASE_DIR = "dataset/blockout"
OBJ_DIR = os.path.join(BASE_DIR, "shape_vhacd")
SEQUENCE_PATH = os.path.join(BASE_DIR, "test_sequence.pt")
ID2NAME_PATH = os.path.join(BASE_DIR, "id2shape.pt")
OUTPUT_JSON = "SECUENCIAL/Pruebas_Blockout/solucion_secuencial.json"

# Parámetros del contenedor
CONTAINER_DIMS = np.array([320, 320, 300]) / 1000.0  # en metros
STEP = 0.02

# Cargar datos del dataset
id2name = torch.load(ID2NAME_PATH, map_location="cpu")
sequence_data = torch.load(SEQUENCE_PATH, map_location="cpu", weights_only=False)

# Extraer los nombres de los .obj de la primera secuencia
secuencia = sequence_data[0]
nombres_shapes = [id2name[int(i)] for i in secuencia]

# Generador de posiciones viables
def generate_positions(container_size, mesh_size, step):
    limits = container_size - mesh_size
    for x in np.arange(0, limits[0] + 1e-6, step):
        for y in np.arange(0, limits[1] + 1e-6, step):
            for z in np.arange(0, limits[2] + 1e-6, step):
                yield np.array([x, y, z])

# Proceso secuencial
placed = []
scene = CollisionManager()

for name in nombres_shapes:
    path = os.path.join(OBJ_DIR, name)
    mesh = trimesh.load(path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()

    mesh_size = mesh.extents
    mesh_offset = mesh.bounds[0]  # esquina mínima

    for pos in generate_positions(CONTAINER_DIMS, mesh_size, STEP):
        test_mesh = mesh.copy()
        tf = np.eye(4)
        tf[:3, 3] = pos - mesh_offset  # compensar el offset
        test_mesh.apply_transform(tf)

        if not scene.in_collision_single(test_mesh):
            scene.add_object(name, test_mesh)
            placed.append({
                "id": name,
                "position_m": list(np.round(pos - mesh_offset, 4))  # guardar la real
            })
            break

# Guardar resultado
with open(OUTPUT_JSON, 'w') as f:
    json.dump(placed, f, indent=4)
print(f"Guardado {len(placed)} objetos en {OUTPUT_JSON}")

# Cálculo de volumen
volumen_total = np.prod(CONTAINER_DIMS)
volumen_usado = 0.0
for p in placed:
    m = trimesh.load(os.path.join(OBJ_DIR, p["id"]), force='mesh')
    if not isinstance(m, trimesh.Trimesh):
        m = m.dump().sum()
    volumen_usado += m.volume

print(f"\n Objetos colocados: {len(placed)}")
print(f"Volumen contenedor: {volumen_total:.6f} m³")
print(f"Volumen usado:     {volumen_usado:.6f} m³")
print(f"Porcentaje lleno:  {(volumen_usado / volumen_total) * 100:.2f}%")
