# REDES NEURONALES

## AUTORES

- [@rociomartin4](https://github.com/rociomartin4), Universidad de Almería
- [@jmvbrocal](https://github.com/jmvbrocal), Universidad de Almería
- [@Alejandro-Bueso](https://github.com/Alejandro-Bueso), Plataforma Solar Almería


## RESUMEN

Este proyecto implementa un modelo de red neuronal para predecir valores de STEC y PFLUX utilizando datos experimentales cargados desde archivos Excel. El código está diseñado para ser usado en Google Colab, donde los archivos de datos se descargan automáticamente desde Google Drive.
Lo más importante de este código es que puede ser utilizado con cualquier otro conjunto de datos siempre y cuando se cambien los datos correspondientes indicados en el código. Los archivos Excel de entrada deben contener las columnas correctas para que el código funcione correctamente.
El flujo del código incluye la descarga de los archivos, el preprocesamiento de los datos (filtrado y normalización), el entrenamiento de una red neuronal utilizando Keras, y la evaluación del modelo con métricas como el RMSE y el R². Además, el modelo utiliza la técnica de EarlyStopping para evitar el sobreajuste.
Este código es flexible y puede adaptarse fácilmente a diferentes conjuntos de datos simplemente modificando las variables de entrada y salida según se indique en el código.
