import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


df_filtered = pd.read_csv('dataset/Reservas_limpio.csv', sep=';')

# Separar las reservas CANCELADAS y COMPLETADAS
reservas_canceladas = df_filtered[df_filtered['Estado'] == 'CANCELADA']
reservas_completadas = df_filtered[df_filtered['Estado'] == 'COMPLETADA']

reservas_canceladas.to_csv('dataset/Reservas_canceladas.csv', sep=';', index=False)
reservas_completadas.to_csv('dataset/Reservas_completadas.csv', sep=';', index=False)

# Analisis de que modelo se pide más dependiendo la ubicación
reservas_completadas = reservas_completadas[reservas_completadas['Modelo'].notna()]
reservas_completadas = reservas_completadas[reservas_completadas['Modelo'].str.strip() != '']

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

# Analisis cancelaciones por provincia
ct = pd.crosstab(df_filtered['Provincia'], df_filtered['Estado'])
ct_norm = ct.div(ct.sum(1), axis=0)  # porcentajes por fila

plt.figure(figsize=(8, 6))
ct_norm.plot(kind='bar', stacked=True, colormap='Set2')
plt.title('Porcentaje de Estados por Provincia')
plt.ylabel('Proporción')
plt.xticks(rotation=45)
plt.legend(title='Estado', loc='upper right')
plt.tight_layout()
plt.show()


# Analisis por precio vs modelo, deberia mantenerse el precio, si aumento analizar porque
top_modelos = df_filtered['Modelo'].value_counts().nlargest(10).index
df_modelos = df_filtered[df_filtered['Modelo'].isin(top_modelos)]

plt.figure(figsize=(10, 6))
sns.boxplot(
    data=df_modelos,
    x='Modelo',
    y='Precio de la publicacion'
)
plt.xticks(rotation=45)
plt.title('Distribución del Precio de la publicacion por Modelo de auto (Top 10)')
plt.ylabel('Precio final (ARS)')
plt.xlabel('Modelo de auto')
plt.tight_layout()
plt.show()


# Analisis por precio vs dias de alquiler, deberia mantenerse el precio, si aumento analizar porque
plt.figure(figsize=(8, 6))
sns.scatterplot(
    data=df_filtered,
    x='Dias de Alquiler',
    y='Precio final',
    alpha=0.6
)
plt.title('Precio de la reserva vs. Días de alquiler')
plt.xlabel('Días de Alquiler')
plt.ylabel('Precio final')
plt.tight_layout()
plt.show()


# --- GRAFICO DE RESERVAS POR MODELOS Y MARCAS---
top_marcas = df_filtered['Marca'].value_counts().head(6).index
df_top = df_filtered[df_filtered['Marca'].isin(top_marcas)].copy()
df_top['Modelo_simple'] = df_top['Modelo'].astype(str).str.split().str[0]

# Calcular el límite máximo para el eje Y para que sea equitativo
max_y = df_top['Modelo_simple'].value_counts().max()
y_lim = ((max_y // 50) + 1) * 50 

fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(18, 8))
axes = axes.flatten()

for i, marca in enumerate(top_marcas):
    ax = axes[i]
    modelos = df_top[df_top['Marca'] == marca]['Modelo_simple'].value_counts().head(5)
    modelos.plot(kind='bar', ax=ax, color='steelblue')
    ax.set_title(f'{marca} - Top modelos')
    ax.set_xlabel('Modelo')
    ax.set_ylabel('Cantidad')
    ax.tick_params(axis='x', rotation=45)
    ax.set_ylim(0, y_lim)
    ax.set_yticks(range(0, y_lim + 1, 50))

plt.tight_layout()
plt.suptitle('Top 5 modelos por cada una de las 6 marcas más reservadas', fontsize=16, y=1.02)
plt.show()