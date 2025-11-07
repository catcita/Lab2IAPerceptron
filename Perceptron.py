import numpy as np

# 1. función para cargar datos del archivo datos.txt
def cargar_datos(filename):
    """ carga entradas (x) y salidas (y) """
    datos = np.loadtxt(filename, delimiter=',')
    # ñas primeras N-1 columnas son entradas (X), la última es la salida (y)
    x = datos[:, :-1]
    y = datos[:, -1]
    return x, y


# 2. Clase Perceptrón
class Perceptron:
    def __init__(self, num_features, tasa_aprendizaje=0.1, iteraciones=10):
        #inicializar pesos aleatoriamente (incluyendo sesgo)
        #inicialización aleatoria pequeña (rand) para evitar ceros
        self.pesos = np.random.rand(num_features) * 0.01 #inicializacion de los pesos con valores aleatorios pequeños para que el aprendizaje
                                                         # pueda comenzar desde un punto de partida neutral
        self.sesgo = np.random.rand() * 0.01 #inicializacion del sesgo (b en la formula) con un valor aleatorio pequeño
        self.tasa = tasa_aprendizaje #fija la tasa de aprendizaje (alpha en la fórmula), que controla la magnitud de los ajustes en cada corrección
        self.iteraciones = iteraciones #número máximo de iteraciones sobre todo el conjutno de datos

    """Función de paso """
    def paso(self, z):
        #Convierte la suma ponderada (z) en una salida binaria:
        #retorna 1 si z >= 0
        #retorna 0 si z < 0
        return np.where(z >= 0, 1, 0)

    """Función de paso de avance"""
    def prediccion(self, x):
        #calcula la salida predicha
        #suma ponderada: (x * w) + b
        linear_output = np.dot(x, self.pesos) + self.sesgo

        #pasa este resultado a la función de activación (paso(self, z)) para obtener la predicción binaria (y_pred)
        return self.paso(linear_output)

    """Función de regla de aprendizaje """
    def train(self, x, y):
        #aplica la regla de aprendizaje del perceptrón
        print(f"Pesos iniciales: {self.pesos}, Sesgo inicial: {self.sesgo:.4f}")

        for iteraciones in range(self.iteraciones):
            errors = 0
            for x_i, y_true in zip(x, y):
                #calcular la predicción
                y_pred = self.prediccion(x_i)

                #calcular el error
                error = y_true - y_pred

                #regla de actualización: ajustar pesos y sesgo si hay error
                if error != 0:
                    # w_nuevo = w_anterior + alpha * error * x_i
                    self.pesos += self.tasa * error * x_i
                    # b_nuevo = b_anterior + alpha * error
                    self.sesgo += self.tasa * error
                    errors += 1

            #imprimir el progreso
            print(f"Época {iteraciones + 1}: Errores = {errors}")

            #condición de convergencia: si no hay errores en una iteración, detener
            if errors == 0:
                print("Convergencia alcanzada.")
                break

        print(f"\nPesos finales: {self.pesos}, Sesgo final: {self.sesgo:.4f}")


# 3. ejecutar el entrenamiento
if __name__ == "__main__":
    x, y = cargar_datos('datos.txt')

    # el número de características de entrada (dos entradas binarias)
    num_features = x.shape[1]

    # crear y entrenar el modelo
    perceptron_model = Perceptron(num_features, tasa_aprendizaje=0.1, iteraciones=20)
    perceptron_model.train(x, y)

    # probar e imprimir en consola la clasificación final
    print("\n--- Pruebas de Clasificación ---")
    for x_i, y_true in zip(x, y):
        prediction = perceptron_model.prediccion(x_i)
        print(f"Entrada: {x_i}, Real: {int(y_true)}, Predicción: {int(prediction)}")