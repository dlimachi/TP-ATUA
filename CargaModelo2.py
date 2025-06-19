import pandas as pd
import joblib

# Cargar modelo y estructuras auxiliares
modelo = joblib.load('modelo_lightgbm_precio.joblib')
columnas = joblib.load('columnas_lightgbm.joblib')
label_encoder = joblib.load('label_encoder_lightgbm.joblib')

# Crear ejemplo manual (usar los mismos nombres que en entrenamiento)
ejemplo_manual = pd.DataFrame([{
    'Temporada': 'Baja',
    'Provincia': 'Buenos Aires',
    'Modelo': '2008 1.6 FELINE TIPTRONIC L/19',
    'Marca': 'Peugeot',
    'Duracion': 1
}])

# Transformar variables categóricas en dummies
ejemplo_dummies = pd.get_dummies(ejemplo_manual)

# Alinear columnas con el modelo entrenado
ejemplo_dummies = ejemplo_dummies.reindex(columns=columnas, fill_value=0)

# Predecir
pred_encoded = modelo.predict(ejemplo_dummies).round().astype(int)[0]
pred_encoded = min(max(pred_encoded, 0), len(label_encoder.classes_) - 1)
pred_clase = label_encoder.inverse_transform([pred_encoded])[0]

print('Predicción del rango de precio por día:', pred_clase)
