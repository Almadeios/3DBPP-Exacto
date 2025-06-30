# simulated_annealing_kitchen.py
import os
import json
import random
import numpy as np
import torch
import trimesh
from trimesh.collision import CollisionManager

# Rutas
BASE_DIR = "dataset/kitchen"
OBJ_DIR = os.path.join(BASE_DIR, "shape_vhacd")
SEQ_PATH = os.path.join(BASE_DIR, "test_sequence.pt")
ID2NAME_PATH = os.path.join(BASE_DIR, "id2shape.pt")
OUTPUT_PATH = "SA/Pruebas_Kitchen/solucion_sa_kitchen.json"

# Parámetros
CONTAINER_DIMS = np.array([320, 320, 300]) / 1000.0
STEP = 0.02
MAX_ITER = 500
TEMP_INICIAL = 100.0
TEMP_FINAL = 0.1
ALPHA = 0.95  # Tasa de enfriamiento

# Cargar dataset
id2name = torch.load(ID2NAME_PATH, map_location="cpu")
secuencia = torch.load(SEQ_PATH, map_location="cpu", weights_only=False)[0]
nombres_shapes = [id2name[int(i)] for i in secuencia[:50]]  # puedes ajustar a más

# Cargar objetos
objetos = []
for name in nombres_shapes:
    path = os.path.join(OBJ_DIR, name)
    mesh = trimesh.load(path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()
    size = mesh.extents
    offset = mesh.bounds[0]
    objetos.append({
        "id": name,
        "mesh": mesh,
        "size": size,
        "offset": offset
    })

# Heurística de colocación
def generate_positions(container_size, mesh_size, step):
    limits = container_size - mesh_size
    for x in np.arange(0, limits[0] + 1e-6, step):
        for y in np.arange(0, limits[1] + 1e-6, step):
            for z in np.arange(0, limits[2] + 1e-6, step):
                yield np.array([x, y, z])

def insertar_objetos(orden):
    placed = []
    scene = CollisionManager()
    for idx in orden:
        obj = objetos[idx]
        mesh = obj["mesh"]
        size = obj["size"]
        offset = obj["offset"]
        for pos in generate_positions(CONTAINER_DIMS, size, STEP):
            test_mesh = mesh.copy()
            tf = np.eye(4)
            tf[:3, 3] = pos - offset
            test_mesh.apply_transform(tf)
            if not scene.in_collision_single(test_mesh):
                scene.add_object(obj["id"], test_mesh)
                placed.append({
                    "id": obj["id"],
                    "position_m": list(np.round(pos - offset, 4))
                })
                break
    return placed

# Recocido Simulado
orden_actual = list(range(len(objetos)))
random.shuffle(orden_actual)
mejor_sol = insertar_objetos(orden_actual)
mejor_score = len(mejor_sol)
sol_actual = orden_actual
score_actual = mejor_score
temperatura = TEMP_INICIAL

for i in range(MAX_ITER):
    vecino = sol_actual.copy()
    i1, i2 = random.sample(range(len(objetos)), 2)
    vecino[i1], vecino[i2] = vecino[i2], vecino[i1]

    colocados = insertar_objetos(vecino)
    score_nuevo = len(colocados)
    delta = score_nuevo - score_actual

    if delta > 0 or random.random() < np.exp(delta / temperatura):
        sol_actual = vecino
        score_actual = score_nuevo
        if score_nuevo > mejor_score:
            mejor_score = score_nuevo
            mejor_sol = colocados
            print(f"[{i}] Mejorado: {mejor_score} objetos")
        else:
            print(f"[{i}] ↪ Aceptado peor: {score_nuevo}")
    else:
        print(f"[{i}] Rechazado: {score_nuevo}")

    temperatura *= ALPHA
    if temperatura < TEMP_FINAL:
        print("\n🌡 Temperatura mínima alcanzada. Finalizando.")
        break

# Guardar resultado
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    json.dump(mejor_sol, f, indent=4)

print(f"\n Guardado: {OUTPUT_PATH} con {mejor_score} objetos")
