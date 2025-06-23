from ortools.sat.python import cp_model
import json
import itertools

# Cargar objetos
with open('Blockout_items.json', 'r') as f:
    items = json.load(f)

# Dimensiones del contenedor (en metros)
CONTAINER_DIMS = (0.32, 0.32, 0.30)

# Posibles rotaciones ortogonales (permuta de ejes)
ROTATIONS = list(set(itertools.permutations([0, 1, 2])))  # 6 combinaciones

model = cp_model.CpModel()

# Variables para cada item
o_vars = []      # objeto es colocado (bool)
pos_vars = []    # (x, y, z) posiciones
rot_vars = []    # rotación elegida (0..5)

scale = 1000  # Para trabajar en milímetros como enteros

container = tuple(int(d * scale) for d in CONTAINER_DIMS)

for idx, item in enumerate(items):
    dims = [int(d * scale) for d in item['dims']]

    o = model.NewBoolVar(f"use_{idx}")
    r = model.NewIntVar(0, len(ROTATIONS) - 1, f"rot_{idx}")

    x = model.NewIntVar(0, container[0], f"x_{idx}")
    y = model.NewIntVar(0, container[1], f"y_{idx}")
    z = model.NewIntVar(0, container[2], f"z_{idx}")

    o_vars.append(o)
    rot_vars.append(r)
    pos_vars.append((x, y, z))

    # Restricción: el objeto rotado debe caber dentro del contenedor
    for rot_id, rot in enumerate(ROTATIONS):
        dim_rot = [dims[rot[i]] for i in range(3)]
        fits_x = model.NewBoolVar(f"fits_x_{idx}_{rot_id}")
        fits_y = model.NewBoolVar(f"fits_y_{idx}_{rot_id}")
        fits_z = model.NewBoolVar(f"fits_z_{idx}_{rot_id}")

        model.Add(x + dim_rot[0] <= container[0]).OnlyEnforceIf(fits_x)
        model.Add(y + dim_rot[1] <= container[1]).OnlyEnforceIf(fits_y)
        model.Add(z + dim_rot[2] <= container[2]).OnlyEnforceIf(fits_z)

        # Crear variable booleana: r == rot_id
        r_is_rot_id = model.NewBoolVar(f"r_eq_{idx}_{rot_id}")
        model.Add(r == rot_id).OnlyEnforceIf(r_is_rot_id)
        model.Add(r != rot_id).OnlyEnforceIf(r_is_rot_id.Not())

        # EnforceAnd solo si el objeto está activo y en esa rotación
        model.AddBoolAnd([fits_x, fits_y, fits_z]).OnlyEnforceIf([o, r_is_rot_id])

# Restricción: no colisionar
for i in range(len(items)):
    for j in range(i + 1, len(items)):
        bi = o_vars[i]
        bj = o_vars[j]

        xi, yi, zi = pos_vars[i]
        xj, yj, zj = pos_vars[j]

        no_overlap = []
        for rot_i in range(len(ROTATIONS)):
            for rot_j in range(len(ROTATIONS)):
                dims_i = [int(items[i]['dims'][ROTATIONS[rot_i][k]] * scale) for k in range(3)]
                dims_j = [int(items[j]['dims'][ROTATIONS[rot_j][k]] * scale) for k in range(3)]

                conds = [
                    xi + dims_i[0] <= xj,
                    xj + dims_j[0] <= xi,
                    yi + dims_i[1] <= yj,
                    yj + dims_j[1] <= yi,
                    zi + dims_i[2] <= zj,
                    zj + dims_j[2] <= zi,
                ]
                bools = [model.NewBoolVar(f"disj_{i}_{j}_{k}") for k in range(6)]
                for k in range(6):
                    model.Add(conds[k]).OnlyEnforceIf(bools[k])

                no_overlap.append(model.NewBoolVar(f"no_overlap_{i}_{j}_{rot_i}_{rot_j}"))
                model.AddBoolOr(bools).OnlyEnforceIf(no_overlap[-1])
                # Variable booleana: r_i == rot_i
                r_i_eq = model.NewBoolVar(f"r_{i}_is_{rot_i}")
                model.Add(rot_vars[i] == rot_i).OnlyEnforceIf(r_i_eq)
                model.Add(rot_vars[i] != rot_i).OnlyEnforceIf(r_i_eq.Not())

                # Variable booleana: r_j == rot_j
                r_j_eq = model.NewBoolVar(f"r_{j}_is_{rot_j}")
                model.Add(rot_vars[j] == rot_j).OnlyEnforceIf(r_j_eq)
                model.Add(rot_vars[j] != rot_j).OnlyEnforceIf(r_j_eq.Not())

                # Variable auxiliar para conjunción de condiciones
                condition = model.NewBoolVar(f"use_{i}_{j}_{rot_i}_{rot_j}")
                model.AddBoolAnd([bi, bj, r_i_eq, r_j_eq]).OnlyEnforceIf(condition)
                model.AddBoolOr([bi.Not(), bj.Not(), r_i_eq.Not(), r_j_eq.Not()]).OnlyEnforceIf(condition.Not())

                # Si se cumple la condición, debe cumplirse no_overlap
                model.AddImplication(condition, no_overlap[-1])


# Objetos no deben flotar (descansan en piso u otro)
# Simplificado: si z > 0, debe haber otro objeto debajo que lo soporte
# (se puede refinar)

# Objetivo: maximizar el número de objetos colocados
model.Maximize(sum(o_vars))

solver = cp_model.CpSolver()
status = solver.Solve(model)

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("Objetos colocados:")
    for idx, o in enumerate(o_vars):
        if solver.BooleanValue(o):
            pos = tuple(solver.Value(v) for v in pos_vars[idx])
            rot = solver.Value(rot_vars[idx])
            print(f"{items[idx]['id']} en {pos}, rotación {ROTATIONS[rot]}")
else:
    print("No se encontró solución.")


placed = []

if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
    print("Objetos colocados:")
    for idx, o in enumerate(o_vars):
        if solver.BooleanValue(o):
            pos = tuple(solver.Value(v) for v in pos_vars[idx])
            rot = solver.Value(rot_vars[idx])
            dims = items[idx]['dims']
            dims_rot = [dims[ROTATIONS[rot][i]] for i in range(3)]
            print(f"{items[idx]['id']} en {pos}, rotación {ROTATIONS[rot]}")
            name = items[idx]['id']
            shape = None
            for key in ['Z', 'L', 'T', 'I', 'B']:
                if f"_{key}_" in name:
                    shape = key
                    break

            placed.append({
                "id": name,
                "position_mm": [int(p) for p in pos],
                "dimensions_mm": [int(d * scale) for d in dims_rot],
                "rotation": ROTATIONS[rot],
                "shape": shape
            })

    # Guardar solución en JSON
    with open("solucion_binpacking.json", "w") as f:
        json.dump(placed, f, indent=4)
    print("Solución guardada en 'solucion_binpacking.json'")
else:
    print("No se encontró solución.")
