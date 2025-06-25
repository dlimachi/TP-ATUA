import streamlit as st
import pandas as pd
import joblib

# Cargar modelo y columnas
modelo = joblib.load('modelo_svm_precio.joblib')
columnas = joblib.load('columnas_entrenamiento.joblib')

st.title("Predicción de rango de precio por día")

# Inputs del usuario
temporada = st.selectbox("Temporada", ['Alta', 'Baja'])
provincia = st.selectbox("Provincia", ['Buenos Aires', 'CABA', 'Catamarca', 'Chaco', 'Chubut', 'Córdoba', 'Corrientes',
    'Entre Ríos', 'Formosa', 'Jujuy', 'La Pampa', 'La Rioja', 'Mendoza', 'Misiones',
    'Neuquén', 'Río Negro', 'Salta', 'San Juan', 'San Luis', 'Santa Cruz',
    'Santa Fe', 'Santiago del Estero', 'Tierra del Fuego', 'Tucumán'])
modelo_auto = st.text_input("Modelo del auto", "Onix")
marca = st.selectbox("Marca", [
    'Peugeot', 'Volkswagen', 'Chevrolet', 'Fiat', 'Renault', 'Ford', 'Toyota', 'Nissan',
    'Honda', 'Citroen', 'Hyundai', 'Kia', 'Mercedes', 'BMW', 'Audi', 'Jeep', 'Chery'
])
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
