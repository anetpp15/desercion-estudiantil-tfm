# Comparación de algoritmos de Machine Learning con interpretabilidad SHAP para la predicción temprana de deserción estudiantil en educación superior

Este repositorio contiene el código del Trabajo de Fin de Máster (TFM) dedicado a comparar algoritmos de aprendizaje automático, con interpretabilidad SHAP, para predecir la deserción estudiantil en educación superior.

## Contenido del repositorio

| Fichero | Descripción |
|---|---|
| `DesercionEstudiantil_TFM_final.ipynb` | Cuaderno con el pipeline completo: carga y depuración de datos, entrenamiento, evaluación e interpretabilidad de los modelos. |
| `figuras_memoria.py` | Script que genera las figuras incluidas en la memoria a partir de los resultados del cuaderno. |
| `dataset.csv` | Copia de respaldo del conjunto de datos original. |

## Datos

El conjunto de datos procede de Kaggle: [Higher Education Predictors of Student Retention](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention). El cuaderno lo descarga automáticamente al ejecutarse, por lo que no es necesario subir ningún archivo.

## Ejecución del proyecto

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/anetpp15/desercion-estudiantil-tfm/blob/main/DesercionEstudiantil_TFM_final.ipynb)

1. Pulsa el botón **Open in Colab**.
2. Ejecuta las celdas en orden, por ejemplo con **Entorno de ejecución > Ejecutar todas**.
3. Para generar las figuras de la memoria, ejecuta al final del cuaderno una celda con:

   ```python
   !wget -q https://raw.githubusercontent.com/anetpp15/desercion-estudiantil-tfm/main/figuras_memoria.py
   %run -i figuras_memoria.py
   ```

## Tecnologías y librerías

**Entorno:** Google Colaboratory  
**Lenguaje:** Python 3  
**Librerías principales:**
- `pandas` y `numpy`: tratamiento y limpieza de datos
- `matplotlib`: visualización
- `scikit-learn`: modelado predictivo y evaluación
- `shap`: interpretabilidad de los modelos
