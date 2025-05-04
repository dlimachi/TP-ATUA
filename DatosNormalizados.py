import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker


df = pd.read_csv('dataset/Reservas.csv', sep=';')

# Eliminamos columnas no necesarias
df_filtered = df.drop(columns=['Email', 'Dni', 'Telefono', 'uuid', 'Cliente', 'Proveedor de carro', 'Patente', 'Anifitrion', 'Celular', 'Iva', 'Observaciones', 'Pago Anfitrion', 'Condicion', 'id'])

# Normalización de fechas
df_filtered['Fecha de creacion'] = pd.to_datetime(df_filtered['Fecha de creacion'], errors='coerce')
fecha_limite = pd.to_datetime('2024-12-01')
df_filtered = df_filtered[df_filtered['Fecha de creacion'] >= fecha_limite]


#Normalización de Ubicación
df_filtered['Provincia'] = df_filtered['Ubicacion'].str.split(',').str[-2].str.strip()
df_filtered['Provincia'] = df_filtered['Provincia'].str.replace('Province', '', regex=False).str.strip()
df_filtered['Provincia'] = df_filtered['Provincia'].str.lower().str.strip()
df_filtered['Provincia'] = df_filtered['Provincia'].replace({
   'capital federal': 'buenos aires',
    'ciudad autónoma de buenos aires': 'buenos aires',
    'caba': 'buenos aires',
    'caballito' : 'buenos aires',
    'córdoba capital': 'córdoba',
    'aeropuerto' : 'rio negro'
})
df_filtered['Provincia'] = df_filtered['Provincia'].str.title()


#Normalización de Precios
columnas_monetarias = [
    'Nuevo Precio', 'Precio de la publicacion', 'Precio de la reserva',
    'Gastos administrativo', 'Seguro base', 'Seguro Contra Terceros',
    'Seguro Premium', 'Precio final', 'Pago seña',
    'Pendiente Por Cobrar', 'Monto de la garantia'
]

for col in columnas_monetarias:
    df_filtered[col] = (
        df_filtered[col]
        .astype(str)
        .str.replace('ARS', '', regex=False)
        .str.replace('.', '', regex=False)
        .str.replace(',', '', regex=False)
        .str.strip()
        .replace('', '0')  # en caso de que haya campos vacíos
        .astype(float)
    )


## REVISAR QUE TODOS LOS PRECIOS NINGUNO TENGA 0 O NULL? O SOLO PRECIO FINAL?
df_filtered = df_filtered[
    (df_filtered['Precio final'].notna()) & 
    (df_filtered['Precio final'] > 0)
]

df_filtered.to_csv('dataset/Reservas_limpio.csv', sep=';', index=False)

