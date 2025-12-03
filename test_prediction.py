"""
Script de prueba para verificar que el modelo funciona correctamente
Carga una muestra del dataset de entrenamiento y hace predicciones
"""
import numpy as np
import pandas as pd
from predict_expression import predict_expression, model, model_info

print("="*60)
print("🧪 PRUEBA DEL MODELO CON DATOS REALES DEL DATASET")
print("="*60)

# Cargar el dataset
data = pd.read_csv('grammatical_facial_expressions.csv')

def parse_landmarks(landmark_str):
    """Parsea el string de landmarks"""
    parts = str(landmark_str).split()
    numeric_values = []
    for part in parts:
        try:
            numeric_values.append(float(part))
        except ValueError:
            continue
    return numeric_values

# Preparar datos
data_clean = data[data['target'].notna()].copy()
landmarks_parsed = data_clean['0'].apply(parse_landmarks)
lengths = landmarks_parsed.apply(len)
mask_valid = lengths == 301
data_test = data_clean[mask_valid].head(20).copy()  # Tomar 20 muestras de prueba

print(f"\n📊 Probando con {len(data_test)} muestras del dataset...\n")

correctas = 0
for idx, row in data_test.iterrows():
    # Extraer vector de características
    vector = parse_landmarks(row['0'])
    real = int(row['target'])
    expresion = row['expression']
    
    # Hacer predicción
    pred = predict_expression(vector)
    
    correcto = "✅" if pred == real else "❌"
    if pred == real:
        correctas += 1
    
    print(f"{correcto} Expresión: {expresion:15s} | Real: {real} | Predicho: {pred}")

accuracy = correctas / len(data_test)
print(f"\n{'='*60}")
print(f"✅ Accuracy en muestras de prueba: {accuracy*100:.1f}% ({correctas}/{len(data_test)})")
print(f"{'='*60}")

if accuracy > 0.8:
    print("\n💡 El modelo funciona bien con datos del dataset.")
    print("   Si no funciona en tiempo real, el problema está en la extracción")
    print("   de landmarks desde la cámara web.")
else:
    print("\n⚠️ El modelo no está prediciendo bien. Verificar:")
    print("   1. Que el modelo esté correctamente guardado")
    print("   2. Que las características coincidan")
