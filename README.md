#Lab2IAPerceptron: Implementación del Algoritmo Perceptrón

Este proyecto contiene una implementación del algoritmo de aprendizaje del **Perceptrón** desde cero, 
utilizando Python y NumPy. El objetivo del proyecto es entrenar este modelo para resolver un problema 
de clasificación binaria simple, como la compuerta lógica **AND**.

##¿Qué es el Perceptrón?

El Perceptrón es el algoritmo de aprendizaje supervisado más elemental para la **clasificación binaria** 
(salidas 0 o 1). Fue desarrollado por Frank Rosenblatt en 1957.

El modelo se basa en la **regla de aprendizaje del Perceptrón**, que ajusta los pesos ($\mathbf{w}$) 
y el sesgo ($b$) cada vez que el modelo comete un error de clasificación, garantizando la convergencia 
si los datos son linealmente separables.

La actualización se realiza de la siguiente manera:

$$
\mathbf{w}_{nuevo} = \mathbf{w}_{anterior} + \alpha \cdot (y_{verdadero} - y_{pred}) \cdot \mathbf{x}
$$
$$
b_{nuevo} = b_{anterior} + \alpha \cdot (y_{verdadero} - y_{pred})
$$

Donde $\alpha$ es la tasa de aprendizaje.

## Estructura del Proyecto

| Archivo/Carpeta | Descripción |
| :--- | :--- |
| `Perceptron.py` | Contiene la lógica principal del modelo Perceptrón, incluyendo la inicialización, la función de activación (paso), la predicción y el algoritmo de entrenamiento. |
| `datos.txt` | Archivo CSV con el dataset de entrenamiento. Actualmente, simula la compuerta lógica **AND** con dos entradas binarias y una salida binaria. |
| `README.md` | Este archivo. |

## Requisitos

Para ejecutar el programa, solo necesitas tener instalado **Python** y la librería **NumPy**.

```bash
pip install numpy