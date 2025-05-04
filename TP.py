import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('dataset/Reservas.csv', sep=';')

# Eliminamos columnas no necesarias
df_cleaned = df.drop(columns=['Email', 'Dni', 'Telefono', 'uuid', 'Cliente', 'Proveedor de carro', 'Patente', 'Anifitrion', 'Celular', 'Iva', 'Observaciones', 'Pago Anfitrion', 'Condicion', 'id'])

# Filtramos fechas posteriores al 01/12/2024
df_cleaned['Fecha de creacion'] = pd.to_datetime(df_cleaned['Fecha de creacion'], errors='coerce')
fecha_limite = pd.to_datetime('2024-12-01')
df_filtered = df_cleaned[df_cleaned['Fecha de creacion'] >= fecha_limite]
df_filtered.to_csv('dataset/Reservas_limpio.csv', sep=';', index=False)

# Separar las reservas CANCELADAS y COMPLETADAS
reservas_canceladas = df_filtered[df_filtered['Estado'] == 'CANCELADA']
reservas_completadas = df_filtered[df_filtered['Estado'] == 'COMPLETADA']

reservas_canceladas.to_csv('dataset/Reservas_canceladas.csv', sep=';', index=False)
reservas_completadas.to_csv('dataset/Reservas_completadas.csv', sep=';', index=False)

# Analisis de que modelo se pide más dependiendo la ubicación
reservas_completadas = reservas_completadas[reservas_completadas['Modelo'].notna()]
reservas_completadas = reservas_completadas[reservas_completadas['Modelo'].str.strip() != '']

reservas_completadas['Provincia'] = reservas_completadas['Ubicacion'].str.split(',').str[-2].str.strip()
# Eliminar la palabra "Province"
reservas_completadas['Provincia'] = reservas_completadas['Provincia'].str.replace('Province', '', regex=False).str.strip()
# Cuenta como normalizar el pasar a minuscula?
reservas_completadas['Provincia'] = reservas_completadas['Provincia'].str.lower().str.strip()
reservas_completadas['Provincia'] = reservas_completadas['Provincia'].replace({
   'capital federal': 'buenos aires',
    'ciudad autónoma de buenos aires': 'buenos aires',
    'caba': 'buenos aires',
    'caballito' : 'buenos aires',
    'córdoba capital': 'córdoba'
})
reservas_completadas['Provincia'] = reservas_completadas['Provincia'].str.title()


modelo_por_ubicacion = reservas_completadas.groupby(['Provincia', 'Modelo']).size().reset_index(name='Cantidad')
modelo_por_ubicacion = modelo_por_ubicacion.sort_values('Cantidad', ascending=False)

conteo_provincias = reservas_completadas['Provincia'].value_counts()
provincias_filtradas = conteo_provincias[conteo_provincias > 10].index
reservas_filtradas = reservas_completadas[reservas_completadas['Provincia'].isin(provincias_filtradas)]

reservas_filtradas = reservas_filtradas[reservas_filtradas['Modelo'].notna()]

# Grafico
plt.figure(figsize=(16, 8))
sns.countplot(data=reservas_filtradas, x='Provincia', hue='Modelo', dodge=True)
plt.xticks(rotation=45)
plt.title("Modelos más pedidos por Provincia")
plt.xlabel("Provincia")
plt.ylabel("Cantidad")
plt.legend(title='Modelo', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()