import trimesh
import json
import time
import os

# Ruta al archivo JSON con los objetos
JSON_PATH = 'Blockout_items.json'
# Ruta a la carpeta donde están los .obj
OBJ_DIR = 'blockout'

# Cargar el JSON con la lista de objetos
with open(JSON_PATH, 'r') as f:
    items = json.load(f)

# Mostrar cada objeto durante 2 segundos
for item in items:
    obj_path = os.path.join(OBJ_DIR, item["id"])
    print(f"Mostrando: {item['id']}")

    try:
        mesh = trimesh.load(obj_path, force='mesh')
        mesh.show()
        time.sleep(2)  # Esperar 2 segundos
    except Exception as e:
        print(f"Error al cargar {item['id']}: {e}")
