import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
import joblib

# Cargar datos
df = pd.read_csv('dataset/Reservas_transformado.csv', sep=';')

df = df[df['Estado'] == 'COMPLETADA']

df.to_csv('dataset/Reservas_transformado.csv', sep=';', index=False)

# Variables predictoras
features = ['Temporada', 'Provincia', 'Modelo', 'Marca', 'Duracion']
X = pd.get_dummies(df[features])

# Variable target
y = df['Rango de precio']

# Separar en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definir la grilla de hiperparámetros
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}

# Entrenar SVM con validación cruzada
svm = SVC(probability=True, random_state=42)
grid_svm = GridSearchCV(svm, param_grid_svm, cv=5)
grid_svm.fit(X_train, y_train)

# Mostrar resultados
print('Mejores hiperparámetros para SVM:', grid_svm.best_params_)
print('Accuracy en test:', grid_svm.score(X_test, y_test))

y_pred = grid_svm.predict(X_test)
print("\nMatriz de Confusión:")
print(confusion_matrix(y_test, y_pred))

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred))

# Guardar el mejor modelo
mejor_modelo = grid_svm.best_estimator_
joblib.dump(mejor_modelo, 'modelo_svm_precio.joblib')

# Guardar columnas usadas en entrenamiento
joblib.dump(X.columns.tolist(), 'columnas_entrenamiento.joblib')
