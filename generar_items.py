import trimesh
import os
import json

INPUT_DIR = "blockout"
OUTPUT_JSON = "items.json"

items = []

for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".obj"):
        path = os.path.join(INPUT_DIR, filename)
        mesh = trimesh.load(path, force='mesh')
        bounds = mesh.bounding_box.bounds
        size = bounds[1] - bounds[0]  # Largo, Ancho, Alto

        items.append({
            "id": filename,
            "dims": size.tolist()
        })

with open(OUTPUT_JSON, "w") as f:
    json.dump(items, f, indent=4)

print(f"Guardado: {OUTPUT_JSON}")
