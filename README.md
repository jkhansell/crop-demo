# 🌾 Clasificación de Cultivos con Machine Learning

## Descripción del Problema

Este proyecto aborda un problema de **Machine Learning supervisado** cuyo objetivo es predecir el tipo de cultivo más adecuado dadas ciertas condiciones ambientales y del suelo.

Se trata de un problema de **clasificación multiclase**, donde el modelo aprende a asociar características agrícolas con el cultivo que mejor se adapta a dichas condiciones.

### Problema de Investigación

¿Es posible predecir de manera confiable el cultivo más adecuado para una región agrícola utilizando únicamente variables ambientales y químicas del suelo, tales como nutrientes (N, P, K), temperatura, humedad, pH y precipitación?

Más específicamente, se busca determinar si existe una relación aprendible entre las condiciones agroclimáticas observadas y la selección óptima de cultivos mediante modelos de aprendizaje automático.

### Hipótesis

Las condiciones ambientales y las propiedades químicas del suelo contienen suficiente información para distinguir patrones característicos entre distintos cultivos, permitiendo que un modelo de clasificación supervisada alcance un desempeño predictivo significativamente superior al azar.

En particular, se plantea que:

- diferentes cultivos presentan rangos óptimos diferenciables de temperatura, humedad, pH y precipitación,
- estas diferencias generan fronteras de decisión que pueden ser aprendidas por modelos de Machine Learning,
- modelos de clasificación entrenados sobre estas variables podrán generalizar correctamente a nuevas observaciones ambientales.

## Objetivo

Construir un modelo de clasificación capaz de predecir el tipo de cultivo recomendado para ciertas a partir de variables físicas y químicas del entorno agrícola.

---

## Dataset

Realizando una búsqueda en Kaggle encontramos este [dataset](https://www.kaggle.com/datasets/ryandinh/agricultural-production-optimization) que puede aportar a la solución del problem de investigación. 


Cada fila del dataset representa una observación agrícola con mediciones ambientales y del suelo.

### Variables de entrada (features)

| Variable | Descripción |
|---|---|
| N | Contenido de nitrógeno en el suelo |
| P | Contenido de fósforo en el suelo |
| K | Contenido de potasio en el suelo |
| temperature | Temperatura promedio (°C) |
| humidity | Humedad relativa (%) |
| ph | Nivel de pH del suelo |
| rainfall | Precipitación acumulada (mm) |

### Variable objetivo (target)

| Variable | Descripción |
|---|---|
| label | Cultivo recomendado |

Ejemplos de clases: *rice, maize, chickpea, mango, banana, coffee, cotton, apple, orange, etc.*

## Problema técnico
Materializando el problema podemos decir que este consiste en aprender una función $f(X) \rightarrow L_i$ donde $L_i$ es nuestro tipo de cultivo.

---

### Tipo de Problema

- Aprendizaje supervisado
- Clasificación multiclase
- Variables numéricas continuas
- Salida categórica

---

### Flujo de Machine Learning

El pipeline típico incluye:

1. **Carga de datos**
2. **Análisis exploratorio básico**
3. **Separación entrenamiento / prueba**
4. **Preprocesamiento**
   - escalamiento de variables numéricas
5. **Entrenamiento del modelo**
6. **Evaluación del desempeño**
7. **Predicción sobre nuevos datos**

---

### Modelos de Clasificación

El problema puede abordarse con distintos algoritmos de clasificación, por ejemplo:

- Logistic Regression (baseline)
- Random Forest Classifier
- Support Vector Machines (SVM)
- K-Nearest Neighbors (KNN)
- Gradient Boosting

Cada modelo aprende diferentes tipos de relaciones entre las variables ambientales y el cultivo objetivo.

---

### Métricas de Evaluación

Para evaluar el desempeño del modelo se utilizan métricas estándar de clasificación:

- **Accuracy**: proporción de predicciones correctas.
- **F1-score (macro)**: balance entre precisión y recall considerando todas las clases.
- **Matriz de confusión** (opcional): permite analizar errores por clase.

Debido a que existen múltiples cultivos, el F1-score macro es especialmente útil para evaluar desempeño equilibrado.

---

### Consideraciones del Modelo

- Las variables tienen diferentes escalas físicas, por lo que el escalamiento suele mejorar el desempeño en modelos lineales.
- El problema presenta múltiples clases con patrones climáticos distintos.
- Algunos cultivos comparten condiciones similares, lo que puede generar fronteras de decisión complejas.

---

## Resultado Esperado

Un modelo entrenado debe ser capaz de recibir nuevas condiciones ambientales como entrada:

```X = [N, P, K, temperature, humidity, ph, rainfall]```


y predecir el cultivo más adecuado para esas condiciones.

---
 