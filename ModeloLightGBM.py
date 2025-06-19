import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMRegressor
from sklearn.metrics import classification_report

df = pd.read_csv('dataset/Reservas_transformado.csv', sep=';')

df = df[df['Estado'] == 'COMPLETADA']

# Definir variables predictoras y target
features = ['Temporada', 'Provincia', 'Modelo', 'Marca', 'Duracion']
X = pd.get_dummies(df[features])
y = df['Rango de precio']

joblib.dump(X.columns.tolist(), 'columnas_lightgbm.joblib')

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear modelo LightGBM
lgbm = LGBMRegressor(random_state=42)

# Como el target es categórico, lo convertimos a números para LightGBM
le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

lgbm.fit(X_train, y_train_enc)

joblib.dump(lgbm, 'modelo_lightgbm_precio.joblib')
joblib.dump(le, 'label_encoder_lightgbm.joblib')

y_pred_enc = lgbm.predict(X_test).round().astype(int)
y_pred_enc = [min(max(p, 0), len(le.classes_) - 1) for p in y_pred_enc] 
y_pred = le.inverse_transform(y_pred_enc)

print(classification_report(y_test, y_pred))
print("Modelo y columnas guardados correctamente.")
