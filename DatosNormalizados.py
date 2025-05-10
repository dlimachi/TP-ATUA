import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker


df = pd.read_csv('dataset/Reservas.csv', sep=';')

# Eliminamos columnas no necesarias
df_filtered = df.drop(columns=['Email', 'Dni', 'Telefono', 'uuid', 'Cliente', 'Proveedor de carro', 
                               'Patente', 'Anifitrion', 'Celular', 'Iva', 'Observaciones', 'Pago Anfitrion', 
                               'Condicion', 'id', 'Entrega Aeropuerto', 'Devolucion Aeropuerto'])

# Normalización de fechas
df_filtered['Fecha de creacion'] = pd.to_datetime(df_filtered['Fecha de creacion'], errors='coerce')
fecha_limite = pd.to_datetime('2024-12-01')
df_filtered = df_filtered[df_filtered['Fecha de creacion'] >= fecha_limite]


#Normalización de Ubicación
ciudades_a_provincias = {
    'San Miguel de Tucumán': 'Tucumán',
    'Córdoba': 'Córdoba',
    'Rosario': 'Santa Fe',
    'Mendoza': 'Mendoza',
    'Capital Federal': 'Buenos Aires',
    'CABA': 'Buenos Aires',
    'Neuquén': 'Neuquén',
    'Salta': 'Salta',
    'Campana' : 'Buenos Aires',
    'Córdoba Capital' : 'Córdoba',
    'Godoy Cruz': 'Mendoza',
    'Villa Urquiza CABA': 'Buenos Aires',
    'Matienzo 1431': 'Mendoza',
    'Rio Negro' : 'Río Negro',
    'San Rafael': 'Mendoza',
    'Capital' : 'Buenos Aires',
    'Monteros': 'Tucumán'
}
provincias_validas = {
    'Buenos Aires', 'CABA', 'Catamarca', 'Chaco', 'Chubut', 'Córdoba', 'Corrientes',
    'Entre Ríos', 'Formosa', 'Jujuy', 'La Pampa', 'La Rioja', 'Mendoza', 'Misiones',
    'Neuquén', 'Río Negro', 'Salta', 'San Juan', 'San Luis', 'Santa Cruz',
    'Santa Fe', 'Santiago del Estero', 'Tierra del Fuego', 'Tucumán'
}

def extraer_provincia(ubicacion):
    if pd.isna(ubicacion):
        return np.nan
    partes = ubicacion.split(',')
    if len(partes) >= 2:
        posible_prov = partes[-2].strip()
    else:
        posible_prov = ubicacion.strip()

    posible_prov = posible_prov.replace('Province', '').strip().lower()

    reemplazos = {
        'capital federal': 'buenos aires',
        'ciudad autónoma de buenos aires': 'buenos aires',
        'caba': 'buenos aires',
        'cordoba capital': 'córdoba',
        'san miguel de tucumán': 'tucumán',
        'caballito': 'buenos aires',
        'córdoba capital': 'córdoba',
        'aeropuerto': 'rio negro'
    }

    posible_prov = reemplazos.get(posible_prov, posible_prov)

    # Buscar en el diccionario ciudades_a_provincias
    provincia = ciudades_a_provincias.get(posible_prov.title(), posible_prov.title())

    # Verificar si es una provincia válida
    if provincia in provincias_validas:
        return provincia
    else:
        return np.nan
    
df_filtered['Provincia'] = df_filtered['Ubicacion'].apply(extraer_provincia)

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
        .astype(int)
    )

df_filtered = df_filtered[df_filtered['Origen'].str.lower() != 'rentlyapp']



df_filtered = df_filtered[
    (df_filtered['Precio final'].notna()) & 
    (df_filtered['Precio final'] > 0)
]


## Normalización de modelos
marcas = [
    'Peugeot', 'Volkswagen', 'Chevrolet', 'Fiat', 'Renault', 'Ford', 'Toyota', 'Nissan',
    'Honda', 'Citroen', 'Hyundai', 'Kia', 'Mercedes', 'BMW', 'Audi', 'Jeep', 'Chery'
]

def normalizar_marca(modelo):
    if pd.isna(modelo):
        return 'Otro'
    
    modelo_lower = modelo.lower()
    
    for marca in marcas:
        if marca.lower() in modelo_lower:
            return marca
    
    if any(p in modelo_lower for p in ['208', '207', '308', '301', '2008', '408']):
        return 'Peugeot'
    if any(p in modelo_lower for p in ['gol', 'up', 'vento', 'fox', 'voyage', 'trend', 'polo', 'passat']):
        return 'Volkswagen'
    if any(p in modelo_lower for p in ['prisma', 'onix', 'cruze', 'corsa', 'spin', 'sonic', 'agile']):
        return 'Chevrolet'
    if any(p in modelo_lower for p in ['uno', 'mobi', 'siena', 'palio', 'cronos', 'argo', 'toro']):
        return 'Fiat'
    if any(p in modelo_lower for p in ['stepway', 'logan', 'sandero', 'kwid', 'clio', 'duster', 'captur']):
        return 'Renault'
    if any(p in modelo_lower for p in ['fiesta', 'focus', 'ka', 'ranger', 'eco']):
        return 'Ford'
    if any(p in modelo_lower for p in ['corolla', 'etios', 'hilux', 'yaris', 'sw4']):
        return 'Toyota'
    if any(p in modelo_lower for p in ['versa', 'march', 'sentra', 'kicks', 'frontier']):
        return 'Nissan'
    if any(p in modelo_lower for p in ['glk', 'sprinter', 'class']):
        return 'Mercedes'
    if any(p in modelo_lower for p in ['c3']):
        return 'Citroen'
    if any(p in modelo_lower for p in ['renegade']):
        return 'Jeep'
    if any(p in modelo_lower for p in ['tucson', 'creta']):
        return 'Hyundai'
    if any(p in modelo_lower for p in ['qq']):
        return 'Chery'
    


df_filtered['Marca'] = df_filtered['Modelo'].apply(normalizar_marca)

df_filtered.to_csv('dataset/Reservas_limpio.csv', sep=';', index=False)

