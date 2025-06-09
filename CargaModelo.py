import joblib
import pandas as pd

# Cargar el modelo y las columnas originales del entrenamiento
modelo = joblib.load('modelo_svm_precio.joblib')
columnas_entrenamiento = joblib.load('columnas_entrenamiento.joblib')

# Crear un ejemplo manual (sin dummies)
nuevo_input = pd.DataFrame([{
    'Temporada': 'Alta',
    'Provincia': 'Mendoza',
    'Modelo': 'Onix',
    'Marca': 'Chevrolet',
    'Duracion': 6
}])

# Convertir a dummies como en entrenamiento
input_dummies = pd.get_dummies(nuevo_input)

# Reindexar para que tenga todas las columnas que el modelo espera
input_dummies = input_dummies.reindex(columns=columnas_entrenamiento, fill_value=0)

# Realizar la predicción
prediccion = modelo.predict(input_dummies)

print('Rango de precio predicho:', prediccion[0])

