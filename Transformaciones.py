import pandas as pd

df_filtered = pd.read_csv('dataset/Reservas_limpio.csv', sep=';')

# Crear la categoría de rango de precios por día
def clasificar_precio_dia(precio):
    if pd.isna(precio):
        return 'Desconocido'
    elif precio < 53000:
        return 'Bajo'
    elif precio <= 63000:
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

# ----------- Temporada (Alta o Baja) -----------

df_filtered['Check-in'] = pd.to_datetime(df_filtered['Check-in'], format='%d/%m/%y %H:%M', errors='coerce')
df_filtered['Check-out'] = pd.to_datetime(df_filtered['Check-out'], format='%d/%m/%y %H:%M', errors='coerce')
df_filtered['Fecha de creacion'] = pd.to_datetime(df_filtered['Fecha de creacion'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')

def definir_temporada(fecha):
    if pd.isna(fecha):
        return 'Desconocida'
    mes = fecha.month
    if mes in [12, 1, 2, 7]:
        return 'Alta'
    return 'Baja'

df_filtered['Temporada'] = df_filtered['Check-in'].apply(definir_temporada)

# ----------- Franja horaria de demanda por Check-in -----------
def franja_horaria(fecha):
    if pd.isna(fecha):
        return 'Desconocida'
    hora = fecha.hour
    if 6 <= hora < 12:
        return 'Mañana'
    elif 12 <= hora < 18:
        return 'Tarde'
    elif 18 <= hora < 24:
        return 'Noche'
    else:
        return 'Madrugada'

df_filtered['Franja Check-in'] = df_filtered['Check-in'].apply(franja_horaria)

# ----------- Franja horaria de creación de reserva (opcional) -----------
df_filtered['Franja Creación'] = df_filtered['Fecha de creacion'].apply(franja_horaria)



# Calcular duración en días (para el modelo SVM)
df_filtered['Duracion'] = (df_filtered['Check-out'] - df_filtered['Check-in']).dt.days
df = df_filtered.dropna(subset=['Check-in', 'Check-out'])


df_filtered.to_csv('dataset/Reservas_transformado.csv', sep=';', index=False)
