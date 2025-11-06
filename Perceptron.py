import numpy as np

# 1. función para cargar datos del archivo datos.txt
def cargar_datos(filename):
    """ Carga entradas (x) y salidas (y) """
    datos = np.loadtxt(filename, delimiter=',')
    # Las primeras N-1 columnas son entradas (X), la última es la salida (y)
    x = datos[:, :-1]
    y = datos[:, -1]
    return x, y


# 2. Clase Perceptrón
class Perceptron:
    def __init__(self, num_features, learning_rate=0.1, epochs=10):
        # Inicializar pesos aleatoriamente (incluyendo sesgo)
        # Inicialización aleatoria pequeña (rand) para evitar ceros
        self.weights = np.random.rand(num_features) * 0.01
        self.bias = np.random.rand() * 0.01
        self.learning_rate = learning_rate
        self.epochs = epochs

    def _step_function(self, z):
        """Función de activación (Función de Paso/Heaviside)."""
        return np.where(z >= 0, 1, 0)

    def predict(self, x):
        """Paso de avance: calcula la salida predicha"""
        # Suma ponderada: (x . w) + b
        linear_output = np.dot(x, self.weights) + self.bias
        return self._step_function(linear_output)

    def train(self, x, y):
        """Aplica la regla de aprendizaje del perceptrón."""
        print(f"Pesos iniciales: {self.weights}, Sesgo inicial: {self.bias:.4f}")

        for epoch in range(self.epochs):
            errors = 0
            for x_i, y_true in zip(x, y):
                # Calcular la predicción
                y_pred = self.predict(x_i)

                # Calcular el error
                error = y_true - y_pred

                # Regla de actualización: Ajustar pesos y sesgo si hay error
                if error != 0:
                    # w_nuevo = w_anterior + alpha * error * x_i
                    self.weights += self.learning_rate * error * x_i
                    # b_nuevo = b_anterior + alpha * error
                    self.bias += self.learning_rate * error
                    errors += 1

            # Imprimir el progreso
            print(f"Época {epoch + 1}: Errores = {errors}")

            # Condición de convergencia: si no hay errores en una época, detener
            if errors == 0:
                print("Convergencia alcanzada.")
                break

        print(f"\nPesos finales: {self.weights}, Sesgo final: {self.bias:.4f}")


# 3. Ejecutar el entrenamiento
if __name__ == "__main__":
    x, y = cargar_datos('datos.txt')

    # El número de características de entrada (dos entradas binarias)
    num_features = x.shape[1]

    # Crear y entrenar el modelo
    perceptron_model = Perceptron(num_features, learning_rate=0.1, epochs=20)
    perceptron_model.train(x, y)

    # Probar la clasificación final
    print("\n--- Pruebas de Clasificación ---")
    for x_i, y_true in zip(x, y):
        prediction = perceptron_model.predict(x_i)
        print(f"Entrada: {x_i}, Real: {int(y_true)}, Predicción: {prediction[0]}")