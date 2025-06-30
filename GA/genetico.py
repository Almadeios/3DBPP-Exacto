# genetico.py (versión mejorada con early stopping, objetos empacados, y 20 objetos)
import os
import json
import random
import numpy as np
import trimesh
from trimesh.collision import CollisionManager
import torch
from tqdm import tqdm

# Rutas
BASE_DIR = "blockout"
OBJ_DIR = os.path.join(BASE_DIR, "shape_vhacd")
SEQ_PATH = os.path.join(BASE_DIR, "test_sequence.pt")
ID2NAME_PATH = os.path.join(BASE_DIR, "id2shape.pt")
OUTPUT_PATH = "GA/solucion_genetico.json"

# Parámetros
CONTAINER_DIMS = np.array([320, 320, 300]) / 1000.0
STEP = 0.04
POBLACION = 30
GENERACIONES = 40
EARLY_STOP = 5 # max generaciones sin mejora
MAX_OBJETOS = 30

# Cargar datos
id2name = torch.load(ID2NAME_PATH, map_location="cpu")
secuencia = torch.load(SEQ_PATH, map_location="cpu", weights_only=False)[0]
nombres_shapes = [id2name[int(i)] for i in secuencia[:MAX_OBJETOS]]

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

# Generador de posiciones
def generate_positions(container_size, mesh_size, step):
    limits = container_size - mesh_size
    for x in np.arange(0, limits[0] + 1e-6, step):
        for y in np.arange(0, limits[1] + 1e-6, step):
            for z in np.arange(0, limits[2] + 1e-6, step):
                yield np.array([x, y, z])

# Evaluador
def evaluar(individuo):
    scene = CollisionManager()
    colocados = 0
    volumen = 0.0
    for idx, pos in enumerate(individuo):
        obj = objetos[idx]
        mesh = obj["mesh"].copy()
        tf = np.eye(4)
        tf[:3, 3] = pos - obj["offset"]
        mesh.apply_transform(tf)
        if scene.in_collision_single(mesh):
            break
        scene.add_object(obj["id"], mesh)
        volumen += mesh.volume
        colocados += 1
    return colocados, volumen

# Crear individuo válido
def crear_individuo():
    scene = CollisionManager()
    individuo = []
    for obj in objetos:
        for pos in generate_positions(CONTAINER_DIMS, obj["size"], STEP):
            mesh = obj["mesh"].copy()
            tf = np.eye(4)
            tf[:3, 3] = pos - obj["offset"]
            mesh.apply_transform(tf)
            if not scene.in_collision_single(mesh):
                scene.add_object(obj["id"], mesh)
                individuo.append(pos)
                break
        else:
            return None
    return individuo

# Generar población
print("🔄 Generando población inicial...")
poblacion = []
with tqdm(total=POBLACION) as pbar:
    while len(poblacion) < POBLACION:
        ind = crear_individuo()
        if ind:
            poblacion.append(ind)
            pbar.update(1)

# Evolución
sin_mejora = 0
mejor_score = (-1, 0.0)
mejor_individuo = None
for g in range(GENERACIONES):
    evaluados = [(evaluar(ind), ind) for ind in poblacion]
    evaluados.sort(reverse=True, key=lambda x: (x[0][0], x[0][1]))
    mejor_gen = evaluados[0][0]
    print(f"[GEN {g}] Objetos colocados: {mejor_gen[0]} | Volumen: {mejor_gen[1]:.6f} m³")

    if mejor_gen > mejor_score:
        mejor_score = mejor_gen
        mejor_individuo = evaluados[0][1]
        sin_mejora = 0
    else:
        sin_mejora += 1
        if sin_mejora >= EARLY_STOP:
            print("\n⚠️ Estancado. Terminando...")
            break

    nueva_pob = [evaluados[0][1]]
    while len(nueva_pob) < POBLACION:
        p1, p2 = random.choices(evaluados[:5], k=2)
        hijo = []
        for pos1, pos2 in zip(p1[1], p2[1]):
            elegido = pos1 if random.random() < 0.5 else pos2
            if random.random() < 0.1:
                elegido = elegido + np.random.uniform(-STEP, STEP, 3)
                elegido = np.clip(elegido, 0, CONTAINER_DIMS)
            hijo.append(elegido)
        nueva_pob.append(hijo)
    poblacion = nueva_pob

# Guardar mejor
salida = []
for idx, pos in enumerate(mejor_individuo):
    salida.append({
        "id": objetos[idx]["id"],
        "position_m": list(np.round(pos - objetos[idx]["offset"], 4))
    })

os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    json.dump(salida, f, indent=4)

print(f"\n📄 Guardado: {OUTPUT_PATH} con {len(salida)} objetos")
