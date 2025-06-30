import torch

# Desactiva la protección de weights_only
test_sequence = torch.load("blockout/test_sequence.pt", weights_only=False)
id2shape = torch.load("blockout/id2shape.pt", weights_only=False)

# Mostrar primera secuencia
secuencia = test_sequence[0]

print(f"Longitud de la secuencia 0: {len(secuencia)}")
print("Primeros 10 elementos:")
for i, shape_id in enumerate(secuencia[:10]):
    print(f"{i}: ID {shape_id} → {id2shape[shape_id]}")
