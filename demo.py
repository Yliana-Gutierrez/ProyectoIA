import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. Simulación de Datos (Para el Propósito del Demo) ---
# En un escenario real, estos datos vendrían de APIs o bases de datos de fútbol.

np.random.seed(42) # Para reproducibilidad

n_partidos = 1000

data = {
    'EloRating_EquipoLocal': np.random.randint(1500, 2000, n_partidos),
    'EloRating_EquipoVisitante': np.random.randint(1500, 2000, n_partidos),
    'FormaReciente_EquipoLocal': np.random.randint(0, 15, n_partidos), # Puntos en últimos 5 partidos
    'FormaReciente_EquipoVisitante': np.random.randint(0, 15, n_partidos),
    'GolesPromedioUltimos5_EquipoLocal': np.round(np.random.uniform(0.5, 3.0, n_partidos), 1),
    'GolesPromedioUltimos5_EquipoVisitante': np.round(np.random.uniform(0.5, 3.0, n_partidos), 1),
    'GolesRecibidosPromedioUltimos5_EquipoLocal': np.round(np.random.uniform(0.5, 3.0, n_partidos), 1),
    'GolesRecibidosPromedioUltimos5_EquipoVisitante': np.round(np.random.uniform(0.5, 3.0, n_partidos), 1),
    'AusenciaJugadorClave_EquipoLocal': np.random.randint(0, 2, n_partidos),
    'AusenciaJugadorClave_EquipoVisitante': np.random.randint(0, 2, n_partidos),
    'PromedioGolesDelanteroClave_EquipoLocal': np.round(np.random.uniform(0.1, 1.5, n_partidos), 1),
    'PromedioGolesDelanteroClave_EquipoVisitante': np.round(np.random.uniform(0.1, 1.5, n_partidos), 1),
    'Localia': 1, # Asumimos todos los partidos tienen un equipo local
    'EstiloAtaqueDefensa_EquipoLocal': np.random.choice(['Ofensivo', 'Equilibrado', 'Defensivo'], n_partidos, p=[0.35, 0.4, 0.25]),
    'EstiloAtaqueDefensa_EquipoVisitante': np.random.choice(['Ofensivo', 'Equilibrado', 'Defensivo'], n_partidos, p=[0.35, 0.4, 0.25]),
}

df = pd.DataFrame(data)

# Creación de la variable objetivo 'Resultado' de forma simplificada
# Generamos resultados con una tendencia: local (45%), empate (25%), visitante (30%)
resultados_simulados = np.random.choice(['Victoria Local', 'Empate', 'Victoria Visitante'], n_partidos, p=[0.45, 0.25, 0.30])
df['Resultado'] = resultados_simulados

print("Primeras 5 filas de los datos simulados:")
print(df.head())
print("\nDistribución de Resultados Simulados:")
print(df['Resultado'].value_counts())

# --- 2. Preprocesamiento de Datos ---

# Codificación de variables categóricas usando One-Hot Encoding
df = pd.get_dummies(df, columns=['EstiloAtaqueDefensa_EquipoLocal', 'EstiloAtaqueDefensa_EquipoVisitante'], drop_first=True)

# Codificación de la variable objetivo
label_encoder = LabelEncoder()
df['Resultado_encoded'] = label_encoder.fit_transform(df['Resultado'])
# Mapeo: {'Empate': 0, 'Victoria Local': 1, 'Victoria Visitante': 2}
# Es importante conocer este mapeo para interpretar las predicciones

print("\nPrimeras 5 filas después del One-Hot Encoding y Label Encoding:")
print(df.head())

# Definir características (X) y variable objetivo (y)
X = df.drop(['Resultado', 'Resultado_encoded'], axis=1)
y = df['Resultado_encoded']

# División de datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # stratify para mantener la proporción de clases

print(f"\nDimensiones de los datos de entrenamiento: {X_train.shape}")
print(f"Dimensiones de los datos de prueba: {X_test.shape}")

# --- 3. Modelo de IA: XGBoost Classifier ---

print("\nEntrenando el modelo XGBoost...")
model = XGBClassifier(objective='multi:softmax',  # Para problemas de clasificación multi-clase
                      num_class=len(label_encoder.classes_), # Número de clases
                      use_label_encoder=False, # Recomendado para XGBoost > 1.3
                      eval_metric='mlogloss', # Métrica de evaluación para clasificación multi-clase
                      n_estimators=100, # Número de árboles
                      learning_rate=0.1, # Tasa de aprendizaje
                      random_state=42)

model.fit(X_train, y_train)
print("Modelo XGBoost entrenado exitosamente.")

# --- 4. Evaluación del Modelo ---

print("\nEvaluando el modelo...")
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Precisión (Accuracy) en el conjunto de prueba: {accuracy:.4f}")

print("\nReporte de Clasificación:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

print("\nMatriz de Confusión:")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicción')
plt.ylabel('Verdadero')
plt.title('Matriz de Confusión')
plt.show()

# --- 5. Demostración de Predicción con un Nuevo Partido (Simulado) ---

print("\n--- Demostración de Predicción para un Nuevo Partido ---")

# Simular datos para un nuevo partido
nuevo_partido_data = {
    'EloRating_EquipoLocal': [1800],
    'EloRating_EquipoVisitante': [1750],
    'FormaReciente_EquipoLocal': [12], # Buen momento
    'FormaReciente_EquipoVisitante': [7], # Momento regular
    'GolesPromedioUltimos5_EquipoLocal': [2.0],
    'GolesPromedioUltimos5_EquipoVisitante': [1.2],
    'GolesRecibidosPromedioUltimos5_EquipoLocal': [0.8],
    'GolesRecibidosPromedioUltimos5_EquipoVisitante': [1.5],
    'AusenciaJugadorClave_EquipoLocal': [0], # Sin ausencias clave
    'AusenciaJugadorClave_EquipoVisitante': [1], # Ausencia clave
    'PromedioGolesDelanteroClave_EquipoLocal': [0.9],
    'PromedioGolesDelanteroClave_EquipoVisitante': [0.5],
    'Localia': [1],
    # Asegúrate de que estas columnas coincidan con las creadas por One-Hot Encoding
    'EstiloAtaqueDefensa_EquipoLocal_Ofensivo': [0], # Ejemplo: No es ofensivo
    'EstiloAtaqueDefensa_EquipoLocal_Equilibrado': [1], # Es equilibrado
    'EstiloAtaqueDefensa_EquipoVisitante_Ofensivo': [0],
    'EstiloAtaqueDefensa_EquipoVisitante_Equilibrado': [0], # Es defensivo
}

# Crear un DataFrame para el nuevo partido
nuevo_partido_df = pd.DataFrame(nuevo_partido_data)

# Asegurarse de que las columnas del nuevo_partido_df coincidan con las de X_train
# Esto es crucial para que el modelo funcione correctamente
# Rellenar con ceros si alguna columna de EstiloAtaqueDefensa no está presente
for col in X.columns:
    if col not in nuevo_partido_df.columns:
        nuevo_partido_df[col] = 0

# Reordenar las columnas para que coincidan con el orden de X_train
nuevo_partido_df = nuevo_partido_df[X_train.columns]

# Realizar la predicción
probabilidades_prediccion = model.predict_proba(nuevo_partido_df)
prediccion_clase = model.predict(nuevo_partido_df)

print("\nProbabilidades de los resultados:")
for i, prob in enumerate(probabilidades_prediccion[0]):
    print(f"  {label_encoder.inverse_transform([i])[0]}: {prob:.2%}")

resultado_final = label_encoder.inverse_transform(prediccion_clase)[0]
print(f"\nEl resultado predicho para el partido es: **{resultado_final}**")