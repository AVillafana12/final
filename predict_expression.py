import numpy as np
import pickle

# Cargar el modelo y el scaler (si es necesario)
with open('modelo_facial_expressions.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler_facial_expressions.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('modelo_info.pkl', 'rb') as f:
    model_info = pickle.load(f)

print(f"✅ Modelo cargado: {model_info['nombre_modelo']}")
print(f"   Accuracy: {model_info['accuracy']:.4f}")
print(f"   Características esperadas: {model_info['num_features']}")
print(f"   Usar scaler: {model_info['usar_scaler']}")

DEBUG = True  # Cambiar a True para ver valores en tiempo real

def predict_expression(landmark_vector):
    """
    Devuelve 0 o 1 según el clasificador.
    
    Parámetros:
        landmark_vector: array numpy o lista con las características de landmarks
                        Debe tener exactamente 301 valores
    
    Return:
        int: 0 o 1 según la predicción
    """
    landmark_vector = np.array(landmark_vector).reshape(1, -1)
    
    # Verificar que tenga el número correcto de características
    if landmark_vector.shape[1] != model_info['num_features']:
        raise ValueError(f"Se esperaban {model_info['num_features']} características, "
                        f"pero se recibieron {landmark_vector.shape[1]}")
    
    if DEBUG:
        print(f"Vector stats - Min: {landmark_vector.min():.3f}, Max: {landmark_vector.max():.3f}, Mean: {landmark_vector.mean():.3f}")
    
    # Aplicar scaler si es necesario
    if model_info['usar_scaler']:
        landmark_vector = scaler.transform(landmark_vector)
    
    # Hacer predicción
    prediction = model.predict(landmark_vector)
    
    # Para Random Forest, podemos obtener la probabilidad
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(landmark_vector)[0]
        if DEBUG:
            print(f"Probabilidades - Clase 0: {proba[0]:.3f}, Clase 1: {proba[1]:.3f}")
    
    return int(prediction[0])
