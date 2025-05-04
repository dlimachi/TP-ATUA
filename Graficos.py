import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import ticker
import math

df_filtered = pd.read_csv('dataset/Reservas_limpio.csv', sep=';')

# --- HISTOGRAMAS DE VARIABLES NUMÉRICAS ---
numeric_columns = df_filtered.select_dtypes(include='number').columns

n = len(numeric_columns)
cols = 3
rows = math.ceil(n / cols)

fig, axes = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 5 * rows))
axes = axes.flatten()

for i, column in enumerate(numeric_columns):
    df_filtered[column].hist(bins=15, ax=axes[i], color='skyblue')
    axes[i].set_title(column)
    axes[i].xaxis.set_major_formatter(
        ticker.FuncFormatter(lambda x, pos: f'{int(x/1000)}k' if x >= 1000 else int(x))
    )

# Quitar subplots vacíos si hay menos columnas que espacios
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.suptitle('Distribución de variables numéricas', fontsize=16)
plt.tight_layout()
plt.show()

# --- GRAFICO DE RESERVAS POR PROVINCIA ---
conteo_provincias = df_filtered['Provincia'].value_counts().sort_values(ascending=False)

plt.figure(figsize=(12, 6))
sns.barplot(x=conteo_provincias.index, y=conteo_provincias.values, color='skyblue')
plt.xticks(rotation=45)
plt.title("Cantidad de autos reservados por provincia")
plt.xlabel("Provincia")
plt.ylabel("Cantidad de reservas")
plt.tight_layout()
plt.show()

# --- GRAFICO DE RESERVAS POR ORIGEN ---
df_filtered['Origen'] = df_filtered['Origen'].fillna('Desconocido')
df_filtered['Origen'] = df_filtered['Origen'].replace('', 'Desconocido')
conteo_origen = df_filtered['Origen'].value_counts()

plt.figure(figsize=(8, 5))
sns.barplot(x=conteo_origen.index, y=conteo_origen.values, color='skyblue')
plt.title('Cantidad de reservas por origen')
plt.xlabel('Origen de reserva')
plt.ylabel('Cantidad')
plt.tight_layout()
plt.show()


# --- MATRIZ
numericas = df_filtered.select_dtypes(include='number')
variancia = numericas.var()
columnas_validas = variancia[variancia > 0].index

columnas_correlacion = [
    'Precio de la reserva',
    'Precio final',
    'Precio de la publicacion',
    'Seguro base',
    'Seguro Contra Tercero',
    'Dias de Alquiler',
    'Monto de la garantia'
]

columnas_correlacion = [col for col in columnas_correlacion if col in numericas.columns]

plt.figure(figsize=(10, 8))
sns.heatmap(numericas[columnas_correlacion].corr(), annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Matriz de correlación de variables seleccionadas")
plt.show()