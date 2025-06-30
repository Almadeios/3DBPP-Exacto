# tabu_search.py
import os
import json
import random
import numpy as np
import trimesh
from trimesh.collision import CollisionManager
import torch

# Rutas
BASE_DIR = "dataset/blockout"
OBJ_DIR = os.path.join(BASE_DIR, "shape_vhacd")
SEQUENCE_PATH = os.path.join(BASE_DIR, "test_sequence.pt")
ID2NAME_PATH = os.path.join(BASE_DIR, "id2shape.pt")
OUTPUT_PATH = "TS/Pruebas_Blockout/solucion_tabu.json"

# Parámetros del contenedor
CONTAINER_DIMS = np.array([320, 320, 300]) / 1000.0
STEP = 0.02
ITERACIONES = 100
MAX_SIN_MEJORA = 10
TABU_TAM = 30

# Cargar datos
id2name = torch.load(ID2NAME_PATH, map_location="cpu")
sequence_data = torch.load(SEQUENCE_PATH, map_location="cpu", weights_only=False)
secuencia = sequence_data[0]
nombres_shapes = [id2name[int(i)] for i in secuencia]

# Cargar objetos
objetos = []
for name in nombres_shapes:
    path = os.path.join(OBJ_DIR, name)
    mesh = trimesh.load(path, force='mesh')
    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump().sum()
    mesh_size = mesh.extents
    mesh_offset = mesh.bounds[0]
    objetos.append({
        "id": name,
        "mesh": mesh,
        "size": mesh_size,
        "offset": mesh_offset
    })

# Generador de posiciones
def generate_positions(container_size, mesh_size, step):
    limits = container_size - mesh_size
    for x in np.arange(0, limits[0] + 1e-6, step):
        for y in np.arange(0, limits[1] + 1e-6, step):
            for z in np.arange(0, limits[2] + 1e-6, step):
                yield np.array([x, y, z])

# Colocador secuencial basado en orden
def insertar_objetos(orden):
    placed = []
    scene = CollisionManager()
    for idx in orden:
        obj = objetos[idx]
        mesh = obj["mesh"]
        size = obj["size"]
        offset = obj["offset"]
        intentos = 0
        for pos in generate_positions(CONTAINER_DIMS, size, STEP):
            intentos += 1
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
        print(f"[INFO] {obj['id']} intentos: {intentos}")
    return placed

# Búsqueda tabú
orden_actual = list(range(len(objetos)))
random.shuffle(orden_actual)
mejor_sol = insertar_objetos(orden_actual)
mejor_score = len(mejor_sol)
tabu = []
sin_mejora = 0

for it in range(ITERACIONES):
    print(f"[IT] Iteración {it} comenzando...")
    vecinos = []
    for _ in range(20):
        v = orden_actual.copy()
        i, j = random.sample(range(len(objetos)), 2)
        v[i], v[j] = v[j], v[i]
        if v not in tabu:
            vecinos.append(v)

    mejor_vecino = orden_actual
    mejor_v_score = -1
    for v in vecinos:
        colocados = insertar_objetos(v)
        score = len(colocados)
        if score > mejor_v_score:
            mejor_v_score = score
            mejor_vecino = v

    if mejor_v_score > mejor_score:
        mejor_sol = insertar_objetos(mejor_vecino)
        mejor_score = mejor_v_score
        sin_mejora = 0
        print(f"[MEJORA] Iteración {it}: {mejor_score} objetos")
    else:
        sin_mejora += 1
        print(f"[NO MEJORA] Iteración {it}: {mejor_v_score} objetos")

    if sin_mejora >= MAX_SIN_MEJORA:
        print("\n🔁 Estancado. Finalizando búsqueda.")
        break

    tabu.append(mejor_vecino)
    if len(tabu) > TABU_TAM:
        tabu.pop(0)
    orden_actual = mejor_vecino

# Guardar resultado
with open(OUTPUT_PATH, 'w') as f:
    json.dump(mejor_sol, f, indent=4)

print(f"\nGuardado: {OUTPUT_PATH} con {len(mejor_sol)} objetos")
