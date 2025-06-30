import os
import json
import random
import numpy as np
import torch
import trimesh
from trimesh.collision import CollisionManager
from copy import deepcopy
from tqdm import tqdm

# Rutas
BASE_DIR = "dataset/blockout"
OBJ_DIR = os.path.join(BASE_DIR, "shape_vhacd")
SEQUENCE_PATH = os.path.join(BASE_DIR, "test_sequence.pt")
ID2NAME_PATH = os.path.join(BASE_DIR, "id2shape.pt")
OUTPUT_JSON = "GRASP/Pruebas_Blockout/solucion_grasp_blockout.json"

# Parámetros del contenedor
CONTAINER_DIMS = np.array([320, 320, 300]) / 1000.0  # en metros
STEP = 0.02
MAX_ITER = 100
CANDIDATOS_GRASP = 5

# Utilidad: posiciones viables
def generate_positions(container_size, mesh_size, step):
    limits = container_size - mesh_size
    for x in np.arange(0, limits[0] + 1e-6, step):
        for y in np.arange(0, limits[1] + 1e-6, step):
            for z in np.arange(0, limits[2] + 1e-6, step):
                yield np.array([x, y, z])

# Cargar datos
id2name = torch.load(ID2NAME_PATH, map_location="cpu")
sequence_data = torch.load(SEQUENCE_PATH, map_location="cpu", weights_only=False)
original_sequence = [id2name[int(i)] for i in sequence_data[0]]

# Evaluar una secuencia de nombres
def evaluar_orden(nombres):
    placed = []
    scene = CollisionManager()

    for name in tqdm(nombres, desc='⏳ Evaluando colocación'):
        path = os.path.join(OBJ_DIR, name)
        mesh = trimesh.load(path, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh):
            mesh = mesh.dump().sum()

        mesh_size = mesh.extents
        mesh_offset = mesh.bounds[0]

        for pos in generate_positions(CONTAINER_DIMS, mesh_size, STEP):
            test_mesh = mesh.copy()
            tf = np.eye(4)
            tf[:3, 3] = pos - mesh_offset
            test_mesh.apply_transform(tf)

            if not scene.in_collision_single(test_mesh):
                scene.add_object(name, test_mesh)
                placed.append({"id": name, "position_m": list(np.round(pos - mesh_offset, 4))})
                break

    return placed

# Cálculo de volumen
def calcular_volumen(placed):
    volumen = 0.0
    for p in placed:
        path = os.path.join(OBJ_DIR, p["id"])
        m = trimesh.load(path, force='mesh')
        if not isinstance(m, trimesh.Trimesh):
            m = m.dump().sum()
        volumen += m.volume
    return volumen

# GRASP principal
mejor_score = -1
mejor_placed = []

for iteracion in tqdm(range(MAX_ITER), desc='🚀 Iteraciones GRASP'):
    for _ in range(CANDIDATOS_GRASP):
        permutado = deepcopy(random.sample(original_sequence, len(original_sequence)))
        resultado = evaluar_orden(permutado)
        volumen = calcular_volumen(resultado)

        if volumen > mejor_score:
            mejor_score = volumen
            mejor_placed = resultado

# Guardar resultado
os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, 'w') as f:
    json.dump(mejor_placed, f, indent=4)

# Reporte
print(f"\nObjetos colocados: {len(mejor_placed)}")
print(f"Volumen contenedor: {np.prod(CONTAINER_DIMS):.6f} m³")
print(f"Volumen usado:     {mejor_score:.6f} m³")
print(f"Porcentaje lleno:  {(mejor_score / np.prod(CONTAINER_DIMS)) * 100:.2f}%")
print(f"Guardado en {OUTPUT_JSON}")
