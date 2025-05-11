import pandas as pd

df_reservas = pd.read_csv("dataset/Reservas_limpio.csv", sep=";")  
df_siniestros = pd.read_csv("dataset/Denuncias-Galicia.csv", sep=";")

df_reservas['id'] = df_reservas['id'].astype(str)
df_siniestros['Número de Reserva'] = df_siniestros['Número de Reserva'].astype(str)

# Unir ambas tablas
df_merged = pd.merge(
    df_reservas, df_siniestros,
    left_on='id', right_on='Número de Reserva',
    how='left',
    suffixes=('', '_siniestro')
)

df_merged['Tuvo incidente'] = df_merged['N Siniestro'].notna().map({True: 'SI', False: 'NO'})

df_merged.drop(columns=['Reintegro Seguro'], inplace=True, errors='ignore')

df_merged['Gasto del Incidente'] = pd.to_numeric(df_merged['Gasto del Incidente'], errors='coerce')
df_merged['Garantía cobrada'] = pd.to_numeric(df_merged['Garantía cobrada'], errors='coerce')

df_merged['Costo neto estimado'] = (
    df_merged['Gasto del Incidente'].fillna(0) - 
    df_merged['Garantía cobrada'].fillna(0)
)

def clasificar_impacto(row):
    if row['Tuvo incidente'] == 'NO':
        return 'Sin incidente'
    elif row['Costo neto estimado'] <= 0:
        return 'Cubierto con garantía'
    else:
        return 'Con pérdida estimada'

df_merged['Impacto incidente'] = df_merged.apply(clasificar_impacto, axis=1)

df_merged.to_csv("reservas_con_incidentes.csv", index=False, sep=";")

print(df_merged['Impacto incidente'].value_counts(dropna=False))

