import streamlit as st
import pandas as pd
import joblib

# Cargar modelo y columnas
modelo = joblib.load('modelo_svm_precio.joblib')
columnas = joblib.load('columnas_entrenamiento.joblib')

st.title("Predicción de rango de precio por día")

modelos_por_marca = {
    'Peugeot': ['208', '207', '308', '301', '2008', '408', 'partner'],
    'Volkswagen': ['gol', 'up', 'vento', 'fox', 'voyage', 'trend', 'polo', 'passat', 'suran', 'nivus', 'amarok'],
    'Chevrolet': ['prisma', 'onix', 'cruze', 'corsa', 'spin', 'sonic', 'agile', 'tracker'],
    'Fiat': ['uno', 'mobi', 'siena', 'palio', 'cronos', 'argo', 'toro', 'pulse'],
    'Renault': ['stepway', 'logan', 'sandero', 'kwid', 'clio', 'duster', 'captur'],
    'Ford': ['fiesta', 'focus', 'ka', 'ranger', 'eco'],
    'Toyota': ['corolla', 'etios', 'hilux', 'yaris', 'sw4', 'hiace'],
    'Nissan': ['versa', 'march', 'sentra', 'kicks', 'frontier'],
    'Mercedes': ['glk', 'sprinter', 'class', 'vito'],
    'Citroen': ['c3'],
    'Jeep': ['renegade'],
    'Hyundai': ['tucson', 'creta'],
    'Chery': ['qq'],
    'Kia': ['cerato'],
    'Audi': ['quattro'],
    'Honda': ['fit']
}

# Inputs del usuario
temporada = st.selectbox("Temporada", ['Alta', 'Baja'])
provincia = st.selectbox("Provincia", ['Buenos Aires', 'CABA', 'Catamarca', 'Chaco', 'Chubut', 'Córdoba', 'Corrientes',
    'Entre Ríos', 'Formosa', 'Jujuy', 'La Pampa', 'La Rioja', 'Mendoza', 'Misiones',
    'Neuquén', 'Río Negro', 'Salta', 'San Juan', 'San Luis', 'Santa Cruz',
    'Santa Fe', 'Santiago del Estero', 'Tierra del Fuego', 'Tucumán'])
marca = st.selectbox("Marca", list(modelos_por_marca.keys()))
modelos_disponibles = modelos_por_marca.get(marca, [])
modelo_auto = st.selectbox("Modelo del auto", modelos_disponibles)
duracion = st.number_input("Duración del alquiler (días)", min_value=1, max_value=30, value=5)

# Armar input como DataFrame
input_manual = pd.DataFrame([{
    'Temporada': temporada,
    'Provincia': provincia,
    'Modelo': modelo_auto,
    'Marca': marca,
    'Duracion': duracion
}])

# Preprocesar como entrenamiento
input_dummies = pd.get_dummies(input_manual)
input_dummies = input_dummies.reindex(columns=columnas, fill_value=0)

# Predicción
if st.button("Predecir"):
    pred = modelo.predict(input_dummies)
    st.success(f"El rango de precio estimado es: **{pred[0]}**")
