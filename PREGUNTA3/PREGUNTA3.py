import csv
import random
import numpy as np
from google.colab import drive

def cxOrdered(ind1, ind2):
    size = min(len(ind1), len(ind2))
    a, b = random.sample(range(size), 2)
    if a > b:
        a, b = b, a

    holes1, holes2 = [True] * size, [True] * size
    for i in range(size):
        if i < a or i > b:
            holes1[ind2[i]] = False
            holes2[ind1[i]] = False

    # We must keep the original values somewhere before scrambling everything
    temp1, temp2 = ind1.copy(), ind2.copy()
    k1, k2 = b + 1, b + 1
    for i in range(size):
        if not holes1[temp1[(i + b + 1) % size]]:
            ind1[k1 % size] = temp1[(i + b + 1) % size]
            k1 += 1

        if not holes2[temp2[(i + b + 1) % size]]:
            ind2[k2 % size] = temp2[(i + b + 1) % size]
            k2 += 1

    # Swap the content between a and b (included)
    for i in range(a, b + 1):
        ind1[i % size], ind2[i % size] = ind2[i % size], ind1[i % size]

    return ind1, ind2

# Montar Google Drive
drive.mount("/content/drive")

# Ruta de tu archivo CSV en Google Drive
archivo = "/content/drive/My Drive/data/grafo_actualizado.csv"

# Cargar la matriz de distancias desde el archivo CSV
with open(archivo, newline='') as file:
    reader = csv.reader(file)
    rows = list(reader)

# Obtener nombres de nodos y distancias
nodes = rows[0][1:]
distances = np.array([[int(cell) for cell in row[1:]] for row in rows[1:]])

# Función de evaluación
def evaluate(individual):
    total_distance = 0
    for i in range(len(individual) - 1):
        total_distance += distances[individual[i]][individual[i+1]]

    # Agregar la distancia desde el último nodo al nodo A
    total_distance += distances[individual[-1]][individual[0]]

    return total_distance

# Configuración del algoritmo genético
population_size = 50
generations = 1000
mutation_probability = 0.1

# Inicialización de la población
population = [random.sample(range(len(nodes)), len(nodes)) for _ in range(population_size)]

# Asegurar que el primer nodo en la secuencia sea el índice de "A"
for ind in population:
    ind.remove(nodes.index("A"))
    ind.insert(0, nodes.index("A"))

# Ejecutar el algoritmo genético
for generation in range(generations):
    # Seleccionar parejas para el cruce
    selected_pairs = [random.sample(population, 2) for _ in range(population_size // 2)]

    # Aplicar cruce y mutación
    for pair in selected_pairs:
        if random.random() < 0.7:
            cxOrdered(pair[0], pair[1])

        if random.random() < mutation_probability:
            index1, index2 = random.sample(range(len(nodes)), 2)
            pair[0][index1], pair[0][index2] = pair[0][index2], pair[0][index1]

            index1, index2 = random.sample(range(len(nodes)), 2)
            pair[1][index1], pair[1][index2] = pair[1][index2], pair[1][index1]

    # Evaluar la población
    fitness_values = [evaluate(ind) for ind in population]

    # Seleccionar la mitad de la población basada en sus valores de aptitud
    selected_indices = np.argsort(fitness_values)[:population_size // 2]
    population = [population[i] for i in selected_indices]

# Obtener el mejor individuo
best_individual = min(population, key=evaluate)

# Imprimir el mejor recorrido comenzando desde "A"
best_route = ["A"] + [nodes[i] for i in best_individual]
print("Mejor recorrido:", best_route)
print("Distancia total:", evaluate(best_individual))