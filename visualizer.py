import json
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import random

# Cargar archivo de solución
with open("solucion_binpacking.json", "r") as f:
    placed = json.load(f)

# Dimensiones del contenedor (en mm)
container_mm = (320, 320, 300)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.set_xlim([0, container_mm[0]])
ax.set_ylim([0, container_mm[1]])
ax.set_zlim([0, container_mm[2]])
ax.set_xlabel('X (mm)')
ax.set_ylabel('Y (mm)')
ax.set_zlabel('Z (mm)')

def draw_box(ax, origin, dims, label):
    x, y, z = origin
    dx, dy, dz = dims
    r = [[x, x + dx], [y, y + dy], [z, z + dz]]
    verts = [
        [r[0][0], r[1][0], r[2][0]],
        [r[0][1], r[1][0], r[2][0]],
        [r[0][1], r[1][1], r[2][0]],
        [r[0][0], r[1][1], r[2][0]],
        [r[0][0], r[1][0], r[2][1]],
        [r[0][1], r[1][0], r[2][1]],
        [r[0][1], r[1][1], r[2][1]],
        [r[0][0], r[1][1], r[2][1]]
    ]
    faces = [
        [verts[0], verts[1], verts[2], verts[3]],
        [verts[4], verts[5], verts[6], verts[7]],
        [verts[0], verts[1], verts[5], verts[4]],
        [verts[2], verts[3], verts[7], verts[6]],
        [verts[1], verts[2], verts[6], verts[5]],
        [verts[4], verts[7], verts[3], verts[0]]
    ]
    color = [random.random() for _ in range(3)]
    box = Poly3DCollection(faces, alpha=0.6, facecolor=color, edgecolor='k')
    ax.add_collection3d(box)
    ax.text(x + dx/2, y + dy/2, z + dz/2, label, color='black', fontsize=6, ha='center')

# Dibujar todos los objetos
for item in placed:
    pos = item["position_mm"]
    dims = item["dimensions_mm"]
    label = item["id"].split(".")[0]
    draw_box(ax, pos, dims, label)

plt.title("Solución de Empaquetamiento 3D")
plt.tight_layout()
plt.show()
