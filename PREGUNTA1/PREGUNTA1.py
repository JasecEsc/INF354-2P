import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# Cargar el conjunto de datos iris
iris_data = load_iris()
features = iris_data.data
labels = iris_data.target

# Normalizar los datos
features_normalized = features / np.amax(features, axis=0)

# Convertir las etiquetas a representación one-hot
num_classes = np.max(labels) + 1
one_hot_labels = np.eye(num_classes)[labels]
labels_one_hot = one_hot_labels.astype(int)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(features_normalized, labels_one_hot, test_size=0.2, random_state=42)

# Definir la arquitectura de la red neuronal
input_size = 4
hidden_size = 5
output_size = 3

# Inicializar los pesos de manera aleatoria
weights_input_hidden = np.random.randn(input_size, hidden_size)
weights_hidden_output = np.random.randn(hidden_size, output_size)

# Definir la función de activación sigmoide y su derivada
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

# Entrenamiento de la red neuronal
epochs = 1000
learning_rate = 0.1

# Iteraciones de entrenamiento
for epoch in range(epochs):
    # Pase hacia adelante (Forward pass)
    hidden_layer_input = np.dot(X_train, weights_input_hidden)
    hidden_layer_output = sigmoid(hidden_layer_input)
    output_layer_input = np.dot(hidden_layer_output, weights_hidden_output)
    predicted_output = sigmoid(output_layer_input)

    # Pase hacia atrás (Backward pass)
    output_error = y_train - predicted_output
    output_delta = output_error * sigmoid_derivative(predicted_output)
    hidden_layer_error = output_delta.dot(weights_hidden_output.T)
    hidden_layer_delta = hidden_layer_error * sigmoid_derivative(hidden_layer_output)

    # Actualizar pesos
    weights_hidden_output += hidden_layer_output.T.dot(output_delta) * learning_rate
    weights_input_hidden += X_train.T.dot(hidden_layer_delta) * learning_rate

# Prueba de la red neuronal
hidden_layer_input_test = np.dot(X_test, weights_input_hidden)
hidden_layer_output_test = sigmoid(hidden_layer_input_test)
output_layer_input_test = np.dot(hidden_layer_output_test, weights_hidden_output)
predicted_output_test = sigmoid(output_layer_input_test)

# Convertir las salidas a etiquetas predichas
y_pred = np.argmax(predicted_output_test, axis=1)
y_true = np.argmax(y_test, axis=1)

# Imprimir las etiquetas predichas y reales
print("Etiquetas predichas:", y_pred)
print("Etiquetas reales:", y_true)