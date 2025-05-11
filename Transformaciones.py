import pandas as pd

df_filtered = pd.read_csv('dataset/Reservas_limpio.csv', sep=';')

# Crear la categoría de rango de precios por día
def clasificar_precio_dia(precio):
    if pd.isna(precio):
        return 'Desconocido'
    elif precio < 35000:
        return 'Bajo'
    elif precio <= 55000:
        return 'Medio'
    else:
        return 'Alto'

df_filtered['Rango de precio'] = df_filtered['Precio de la publicacion'].apply(clasificar_precio_dia)
df_filtered['Rango de precio'] = pd.Categorical(
    df_filtered['Rango de precio'],
    categories=['Bajo', 'Medio', 'Alto', 'Desconocido'],
    ordered=True
)

checkin_parsed = pd.to_datetime(df_filtered['Check-in'], errors='coerce', dayfirst=True)
creacion_parsed = pd.to_datetime(df_filtered['Fecha de creacion'], errors='coerce')

df_filtered['Antiguedad dias'] = (checkin_parsed - creacion_parsed).dt.days

# Clasificar en categorías
def clasificar_antiguedad(dias):
    if pd.isna(dias):
        return 'Desconocido'
    if dias <= 7:
        return 'Último momento'
    elif dias <= 14:
        return 'Moderada'
    else:
        return 'Anticipada'

df_filtered['Antigüedad de Reserva'] = df_filtered['Antiguedad dias'].apply(clasificar_antiguedad)

# Contar cuántas reservas hay por categoría
conteo = df_filtered['Antigüedad de Reserva'].value_counts().reset_index()
conteo.columns = ['Tipo de Antigüedad', 'Cantidad']
print(conteo)

conteo = df_filtered['Rango de precio'].value_counts().reset_index()
conteo.columns = ['Tipo de rango de precio', 'Cantidad']
print(conteo)

print("Pago de garantía:")
print(df_filtered['Pago de garantia'].value_counts(dropna=False))

print("\nDevolución Garantia:")
print(df_filtered['Devolución Garantia'].value_counts(dropna=False))

df_filtered.to_csv('dataset/Reservas_transformado.csv', sep=';', index=False)
