# Redes-Neuronales

################################################################################
################################## AUTORES #####################################
################################################################################

- [@rociomartin4](https://github.com/rociomartin4), Universidad de Almería
- [@jmvbrocal](https://github.com/jmvbrocal), Universidad de Almería


################################################################################
############################ LIMPIADOR DE MEMORIA ##############################
################################################################################

%reset -f  # Reinicia el entorno de trabajo para liberar memoria y evitar posibles conflictos


################################################################################
######################## INSTALACION DE PAQUETES NECESARIOS ####################
################################################################################

# Importamos todas las librerías necesarias para el análisis y modelado de datos

import numpy as np  # Biblioteca para manejo de operaciones matemáticas
import pandas as pd  # Biblioteca para manejo de datos en estructuras de tablas (DataFrames)
import gdown  # Librería para descargar archivos desde Google Drive
import matplotlib.pyplot as plt  # Librería para graficar
from sklearn.model_selection import train_test_split  # Para dividir los datos en entrenamiento y validación
from sklearn.metrics import mean_squared_error, r2_score  # Métricas estadísticas para evaluar el modelo
from sklearn.preprocessing import MinMaxScaler  # Para normalizar los datos dentro de un rango especificado
from scipy.stats import linregress  # Función para obtener la regresión lineal entre dos variables
import tensorflow as tf  # Librería para implementar redes neuronales y modelos de machine learning
from tensorflow.keras.models import Sequential  # Modelo secuencial de redes neuronales
from tensorflow.keras.layers import Dense  # Capa completamente conectada en redes neuronales
from tensorflow.keras.callbacks import EarlyStopping  # Técnica para detener el entrenamiento si no hay mejora
from tensorflow.keras import regularizers  # Herramientas para regularización en las redes neuronales


################################################################################
########################### CARGAMOS LOS DATOS #################################
################################################################################

# Especificamos los enlaces para descargar los archivos de Google Drive.
# El ID del archivo es la parte del enlace de Google Drive que se encuentra entre 'd/' y '/view'

file_id_validation = '1RlqzXzKuXF3majIFQ1ITqlPxxxl7xuch'  # ID del archivo de validación
file_id_training = '1BjRir7yGs8vsU1D52n5YjB9HGfYv3ktT'  # ID del archivo de entrenamiento

# Creamos las URLs de descarga directa para los archivos de Google Drive
url_validation = f'https://drive.google.com/uc?id={file_id_validation}'
url_training = f'https://drive.google.com/uc?id={file_id_training}'

# Descargamos los archivos Excel desde Google Drive
gdown.download(url_validation, 'validation_dataset.xlsx', quiet=False)  # Descargamos el archivo de validación
gdown.download(url_training, 'dataset.xlsx', quiet=False)  # Descargamos el archivo de entrenamiento

# Cargamos los datos desde los archivos Excel descargados
df = pd.read_excel('dataset.xlsx', engine='openpyxl')  # Cargamos los datos de entrenamiento en un DataFrame
dfv = pd.read_excel('validation_dataset.xlsx', engine='openpyxl')  # Cargamos los datos de validación

# Verificamos que los datos se han cargado correctamente imprimiendo las primeras filas
print(df.head())  # Imprimimos las primeras filas del conjunto de datos de entrenamiento

# -----------------------------------------------------
# DATOS DE ENTRENAMIENTO
# -----------------------------------------------------
# Filtramos los datos para obtener solo las filas correspondientes al conjunto de entrenamiento
df_training = df[df['ANNsubset'] == 'Training']  # Construimos un DataFrame solo con los datos de entrenamiento 
 # (modificar a conveniencia o incluso suprimir en caso de no tener que distinguir)

# -----------------------------------------------------
# DATOS DE VALIDACIÓN
# -----------------------------------------------------
# Filtramos los datos para obtener solo las filas correspondientes al conjunto de validación
df_validation = df[df['ANNsubset'] == 'Validation']  # Construimos un DataFrame solo con los datos de validación
 # (modificar a conveniencia o incluso suprimir en caso de no tener que distinguir)


################################################################################
###################################### STEC ####################################
################################################################################

# ------------------------------
# 1. SELECCIÓN DE VARIABLES
# ------------------------------
# Para el conjunto de entrenamiento, seleccionamos las columnas de entrada y salida (modificar a conveniencia)
X_train = df_training[['S(g/L)', 'T_cond(°C)', 'T_evap(°C)', 'F(L/h)']].values  # Variables de entrada
y_train = df_training['STEC'].values  # Variable de salida (STEC)

# Para el conjunto de validación, seleccionamos las columnas correspondientes
X_val = df_validation[['S(g/L)', 'T_cond(°C)', 'T_evap(°C)', 'F(L/h)']].values  # Variables de entrada
y_val = df_validation['STEC'].values  # Variable de salida (STEC)

# También seleccionamos las mismas variables para otro conjunto de validación (en el archivo de validación 2)
X_val2 = dfv[['S(g/L)', 'T_cond(°C)', 'T_evap(°C)', 'F(L/h)']].values  # Variables de entrada
y_val2 = dfv['STEC'].values  # Variable de salida (STEC)

# ------------------------------
# 2. NORMALIZACIÓN DE LOS DATOS
# ------------------------------
# Normalizamos los datos de entrada y salida para que estén en el rango [0.1, 0.9]
scaler_X = MinMaxScaler(feature_range=(0.1, 0.9))  # Escalador para las variables de entrada
scaler_y = MinMaxScaler(feature_range=(0.1, 0.9))  # Escalador para las variables de salida

# Aplicamos la normalización a los datos de entrenamiento y validación
X_train = scaler_X.fit_transform(X_train)  # Normalizamos las entradas de entrenamiento
X_val = scaler_X.transform(X_val)  # Normalizamos las entradas de validación
X_val2 = scaler_X.transform(X_val2)  # Normalizamos las entradas del segundo conjunto de validación

y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))  # Normalizamos las salidas de entrenamiento
y_val = scaler_y.transform(y_val.reshape(-1, 1))  # Normalizamos las salidas de validación
y_val2 = scaler_y.transform(y_val2.reshape(-1, 1))  # Normalizamos las salidas del segundo conjunto de validación

# ------------------------------
# 3. MODELO PRELIMINAR PARA DETECTAR RUIDO
# ------------------------------
# Creamos un modelo preliminar de red neuronal secuencial para predecir 'STEC'
model_prelim = Sequential([
    Dense(32, input_dim=4, activation='relu', kernel_initializer='glorot_normal'),  # Capa de entrada con 32 neuronas
    Dense(16, activation='relu', kernel_initializer='glorot_normal'),  # Capa oculta con 16 neuronas
    Dense(8, activation='relu', kernel_initializer='glorot_normal'),  # Capa oculta con 8 neuronas
    Dense(1, activation='linear')  # Capa de salida (predicción de STEC)
])
# Dense(nº neuronas en la capa actual, numero de entradas, funcion de activación de la primera capa, criterio por el que se inicializan los pesos), Dense(...),...,nº de capas que quiera poner

# Compilamos el modelo usando el optimizador 'adam' y la función de pérdida 'mse' (error cuadrático medio)
model_prelim.compile(optimizer='adam', loss='mse')

# Entrenamos el modelo preliminar con los datos de entrenamiento
model_prelim.fit(X_train, y_train, epochs=500, batch_size=8, verbose=0)
# Epoch: numero de epocas de la fase de entrenamiento
# batch_size: número de muestras que el modelo procesará en cada lote antes de actualizar los pesos.
# verbose: controla el nivel de salida del entrenamiento, si es 0 no se muestra nada, si es 1 se muestra un progreso y si es 2 se muestra más información detallada


# ------------------------------
# 4. FILTRADO DE OUTLIERS (RUIDO)
# ------------------------------
# Realizamos predicciones sobre los datos de entrenamiento
y_pred_init = model_prelim.predict(X_train)

# Desnormalizamos las predicciones y los valores reales para obtener los resultados en su escala original
y_true_init = scaler_y.inverse_transform(y_train)
y_pred_inv_init = scaler_y.inverse_transform(y_pred_init)

# Calculamos el error absoluto entre las predicciones y los valores reales
errors = np.abs(y_true_init - y_pred_inv_init)

# Definimos un umbral para considerar qué datos son "outliers" (valores atípicos)
threshold = np.mean(errors) + 2 * np.std(errors)

# Creamos una máscara para filtrar los datos que están por debajo del umbral de error
mask = errors.flatten() < threshold

# Aplicamos la máscara para eliminar los datos con ruido
X_train_filtered = X_train[mask]  # Datos de entrada filtrados
y_train_filtered = y_train[mask]  # Datos de salida filtrados

# Imprimimos el número de datos filtrados
print(f"Filtrados {np.sum(~mask)} datos con ruido excesivo.")

# ------------------------------
# 5. MODELO FINAL OPTIMIZADO
# ------------------------------
# Creamos el modelo final de red neuronal para predecir 'STEC'
model = Sequential([
    Dense(32, input_dim=4, activation='relu', kernel_initializer='glorot_normal'),  # Capa de entrada con 32 neuronas
    Dense(16, activation='relu', kernel_initializer='glorot_normal'),  # Capa oculta con 16 neuronas
    Dense(8, activation='relu', kernel_initializer='glorot_normal'),  # Capa oculta con 8 neuronas
    Dense(1, activation='linear')  # Capa de salida (predicción de STEC)
])

# Compilamos el modelo con el optimizador 'adam' y la función de pérdida 'mse'
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

# Usamos EarlyStopping para evitar el sobreajuste (parar el entrenamiento si la validación no mejora)
early_stop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

# Entrenamos el modelo con los datos filtrados y validación
history = model.fit(X_train_filtered, y_train_filtered,
                    epochs=1000,
                    batch_size=8,
                    verbose=0,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop])

# ------------------------------
# 6. EVALUACIÓN DEL MODELO
# ------------------------------
def evaluate_and_plot(X, y, dataset_name):
    # Realiza una predicción sobre los datos de entrada X usando el modelo entrenado
    y_pred = model.predict(X)

    # Desnormaliza los valores reales (y) usando el scaler, para obtenerlos en su escala original
    y_true = scaler_y.inverse_transform(y)

    # Desnormaliza las predicciones del modelo (y_pred) para obtenerlas en su escala original
    y_pred_inv = scaler_y.inverse_transform(y_pred)

    # Calcula el **Root Mean Squared Error (RMSE)** entre los valores reales y las predicciones
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_inv))

    # Calcula el **R²** (coeficiente de determinación) que indica cuán bien se ajusta el modelo a los datos
    r2 = r2_score(y_true, y_pred_inv)

    # Imprime los resultados del RMSE y R² con un formato de 3 decimales
    print(f"{dataset_name} - STEC - RMSE: {rmse:.3f}, R2: {r2:.3f}")

    # Crea una figura para el gráfico con un tamaño específico
    plt.figure(figsize=(6,5))

    # Dibuja un gráfico de dispersión de los valores reales (y_true) frente a las predicciones (y_pred_inv)
    plt.scatter(y_true, y_pred_inv, alpha=0.7)  # alpha controla la transparencia de los puntos

    # Dibuja una línea roja discontinua que representa la relación perfecta (y_true == y_pred)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')

    # Etiquetas para los ejes X e Y del gráfico
    plt.xlabel('Real STEC')
    plt.ylabel('Predicho STEC')

    # Título del gráfico que incluye el nombre del dataset y el valor de R²
    plt.title(f'{dataset_name}: STEC\nR2={r2:.3f}, p={linregress(y_true.flatten(), y_pred_inv.flatten()).pvalue:.2e}')

    # Ajusta el diseño para que los elementos del gráfico no se solapen
    plt.tight_layout()

    # Muestra el gráfico
    plt.show()

# Llamadas a la función para evaluar y graficar el rendimiento en diferentes datasets
evaluate_and_plot(X_train_filtered, y_train_filtered, "Entrenamiento")  # Evaluación para el conjunto de entrenamiento filtrado
evaluate_and_plot(X_val, y_val, "Validación")  # Evaluación para el conjunto de validación
evaluate_and_plot(X_val2, y_val2, "Validación Tabla 2")  # Evaluación para otro conjunto de validación


# Función para mostrar los pesos y sesgos aprendidos por el modelo
def mostrar_pesos_y_sesgos(model):
    # Configuración para mostrar todas las columnas en Google Colab
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.expand_frame_repr', False)

    # Iteramos sobre cada capa del modelo
    for i, capa in enumerate(model.layers):
        pesos, sesgos = capa.get_weights()

        # Mostramos la matriz de pesos
        df_pesos = pd.DataFrame(pesos)
        print(f"\n{'='*10} Capa {i+1} - Matriz de Pesos (W^{i+1}) {'='*10}")
        print(df_pesos)

        # Mostramos el vector de sesgo
        df_sesgos = pd.DataFrame(sesgos.reshape(1, -1))
        print(f"\n{'='*10} Capa {i+1} - Vector de Sesgo (b^{i+1}) {'='*10}")
        print(df_sesgos)

# Llamada a la función (asegúrate de tener un modelo definido)
mostrar_pesos_y_sesgos(model)


################################################################################
###################################### PFLUX ###################################
################################################################################

# NOTA: los comentarios en esta celda son análogos a los de la predicción del STEC.

# ------------------------------
# 1. Selección de variables
# ------------------------------
# entrenamiento
X_train = df_training[['S(g/L)', 'T_cond(°C)', 'T_evap(°C)', 'F(L/h)']].values
y_train = df_training['P_flux'].values

# validacion
X_val = df_validation[['S(g/L)', 'T_cond(°C)', 'T_evap(°C)', 'F(L/h)']].values
y_val = df_validation['P_flux'].values

# tabla 2 validacion
X_val2 = dfv[['S(g/L)', 'T_cond(°C)', 'T_evap(°C)', 'F(L/h)']].values
y_val2 = dfv['P_flux'].values

# ------------------------------
# 2. Normalización [0.1, 0.9]
# ------------------------------
scaler_X = MinMaxScaler(feature_range=(0.1, 0.9))
scaler_y = MinMaxScaler(feature_range=(0.1, 0.9))

X_train = scaler_X.fit_transform(X_train)
X_val = scaler_X.transform(X_val)
X_val2 = scaler_X.transform(X_val2)

y_train = scaler_y.fit_transform(y_train.reshape(-1, 1))
y_val = scaler_y.transform(y_val.reshape(-1, 1))
y_val2 = scaler_y.transform(y_val2.reshape(-1, 1))

# ------------------------------
# 3. Entrenamiento preliminar para detectar ruido
# ------------------------------
model_prelim = Sequential([
    Dense(32, input_dim=4, activation='relu', kernel_initializer='glorot_normal'),
    Dense(16, activation='relu', kernel_initializer='glorot_normal'),
    Dense(8, activation='relu', kernel_initializer='glorot_normal'),
    Dense(1, activation='linear')
])

# Dense(nº neuronas en la capa actual, numero de entradas, funcion de activación de la primera capa, criterio por el que se inicializan los pesos), Dense(...),...,nº de capas que quiera poner

model_prelim.compile(optimizer='adam', loss='mse')
model_prelim.fit(X_train, y_train, epochs=500, batch_size=8, verbose=0)

#

# ------------------------------
# 4. Filtrado de outliers
# ------------------------------
y_pred_init = model_prelim.predict(X_train)
y_true_init = scaler_y.inverse_transform(y_train)
y_pred_inv_init = scaler_y.inverse_transform(y_pred_init)

errors = np.abs(y_true_init - y_pred_inv_init)
threshold = np.mean(errors) + 2 * np.std(errors)  # Umbral

mask = errors.flatten() < threshold
X_train_filtered = X_train[mask]
y_train_filtered = y_train[mask]

print(f"Filtrados {np.sum(~mask)} datos con ruido excesivo.")

# ------------------------------
# 5. Modelo final optimizado
# ------------------------------
model = Sequential([
    Dense(32, input_dim=4, activation='relu', kernel_initializer='glorot_normal'),
    Dense(16, activation='relu', kernel_initializer='glorot_normal'),
    Dense(8, activation='relu', kernel_initializer='glorot_normal'),
    Dense(1, activation='linear')
])
model.compile(optimizer='adam', loss='mse', metrics=['mse'])

early_stop = EarlyStopping(monitor='val_loss', patience=50, restore_best_weights=True)

history = model.fit(X_train_filtered, y_train_filtered,
                    epochs=1000,
                    batch_size=8,
                    verbose=0,
                    validation_data=(X_val, y_val),
                    callbacks=[early_stop])


# ------------------------------
# 6. Evaluación
# ------------------------------
def evaluate_and_plot(X, y, dataset_name):
    y_pred = model.predict(X)
    y_true = scaler_y.inverse_transform(y)
    y_pred_inv = scaler_y.inverse_transform(y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_inv))
    r2 = r2_score(y_true, y_pred_inv)

    print(f"{dataset_name} - PFLUX - RMSE: {rmse:.3f}, R2: {r2:.3f}")

    plt.figure(figsize=(6,5))
    plt.scatter(y_true, y_pred_inv, alpha=0.7)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--')
    plt.xlabel('Real PFLUX')
    plt.ylabel('Predicho PFLUX')
    plt.title(f'{dataset_name}: PFLUX\nR2={r2:.3f}, p={linregress(y_true.flatten(), y_pred_inv.flatten()).pvalue:.2e}')
    plt.tight_layout()
    plt.show()

evaluate_and_plot(X_train_filtered, y_train_filtered, "Entrenamiento")
evaluate_and_plot(X_val, y_val, "Validación")
evaluate_and_plot(X_val2, y_val2, "Validación Tabla 2")


def mostrar_pesos_y_sesgos(model):
    # Configuración para mostrar todas las columnas en Google Colab
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.expand_frame_repr', False)

    # Iteramos sobre cada capa del modelo
    for i, capa in enumerate(model.layers):
        pesos, sesgos = capa.get_weights()

        # Mostramos la matriz de pesos
        df_pesos = pd.DataFrame(pesos)
        print(f"\n{'='*10} Capa {i+1} - Matriz de Pesos (W^{i+1}) {'='*10}")
        print(df_pesos)

        # Mostramos el vector de sesgo
        df_sesgos = pd.DataFrame(sesgos.reshape(1, -1))
        print(f"\n{'='*10} Capa {i+1} - Vector de Sesgo (b^{i+1}) {'='*10}")
        print(df_sesgos)

# Llamada a la función (asegúrate de tener un modelo definido)
mostrar_pesos_y_sesgos(model)
