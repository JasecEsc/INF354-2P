import csv
import random
import numpy as np
from deap import base, creator, tools, algorithms
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
    temp1, temp2 = ind1, ind2
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

# Crear tipo de individuo y tipo de población para DEAP
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

# Registrar operadores genéticos
toolbox = base.Toolbox()
toolbox.register("indices", random.sample, range(len(nodes)), len(nodes))
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxOrdered)

def mutate_ordered(individual, indpb):
    # Seleccionar dos índices diferentes
    index1, index2 = random.sample(range(len(nodes)), 2)
    # Intercambiar los valores en los dos índices
    individual[index1], individual[index2] = individual[index2], individual[index1]
    # Asegurar que "A" esté en la posición inicial
    if individual[0] != nodes.index("A"):
        index_a = individual.index(nodes.index("A"))
        individual[0], individual[index_a] = individual[index_a], individual[0]
    
    return individual,

toolbox.register("mutate", mutate_ordered, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Función de evaluación
def evaluate(individual):
    total_distance = 0
    for i in range(len(individual) - 1):
        total_distance += distances[individual[i]][individual[i+1]]
    
    # Agregar la distancia desde el último nodo al nodo A
    total_distance += distances[individual[-1]][individual[0]]
    
    return total_distance,


# Configuración del algoritmo genético
population_size = 50
generations = 1000

# Inicialización de la población
population = toolbox.population(n=population_size)

# Asegurar que el primer nodo en la secuencia sea el índice de "A"
for ind in population:
    ind.remove(nodes.index("A"))
    ind.insert(0, nodes.index("A"))

# Configuración de estadísticas
stats = tools.Statistics(lambda ind: ind.fitness.values)
stats.register("min", np.min)

# Ejecutar el algoritmo genético
algorithms.eaMuPlusLambda(population, toolbox, mu=population_size, lambda_=population_size*2,
                          cxpb=0.7, mutpb=0.2, ngen=generations, stats=stats, halloffame=None, verbose=True)

# Obtener el mejor individuo
best_individual = tools.selBest(population, k=1)[0]

# Imprimir el mejor recorrido comenzando desde "A"
best_route = [nodes[i] for i in best_individual]
print("Mejor recorrido:", best_route)
print("Distancia total:", best_individual.fitness.values[0])