import numpy as np

# Índices de los 100 landmarks más relevantes de MediaPipe Face Mesh
# Incluye: contorno facial, cejas, ojos, nariz, boca, iris
DATASET_POINTS = [
    # Contorno de la cara (17 puntos)
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400,
    # Ceja derecha (5 puntos)
    70, 63, 105, 66, 107,
    # Ceja izquierda (5 puntos)  
    336, 296, 334, 293, 300,
    # Ojos (16 puntos - ojo derecho e izquierdo)
    33, 7, 163, 144, 145, 153, 154, 155,  # Ojo derecho
    263, 249, 390, 373, 374, 380, 381, 382,  # Ojo izquierdo
    # Nariz (9 puntos)
    1, 2, 98, 327, 294, 280, 425, 411, 376,
    # Boca exterior (12 puntos)
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375,
    # Boca interior (8 puntos)
    78, 191, 80, 81, 82, 13, 312, 311,
    # Iris derecho (5 puntos)
    468, 469, 470, 471, 472,
    # Iris izquierdo (5 puntos)
    473, 474, 475, 476, 477,
    # Puntos adicionales para completar 100 (18 puntos)
    234, 127, 162, 21, 54, 103, 67, 109, 10, 338, 151, 337, 299, 333, 298, 301, 368, 264
]

# Verificar que tengamos exactamente 100 puntos
assert len(DATASET_POINTS) == 100, f"Se necesitan 100 puntos, pero hay {len(DATASET_POINTS)}"

def extract_landmarks(landmarks, frame_width, frame_height, normalize=False):
    """
    Convierte los landmarks de MediaPipe en un vector de 301 valores.
    IMPORTANTE: El dataset usa VALORES ABSOLUTOS en píxeles, NO normalizados.
    
    Estructura del vector:
    - 1 valor inicial (timestamp/frame_id) - Usamos un valor promedio del dataset (~1.39e9)
    - 100 puntos × 3 coords (x, y, z) = 300 valores en PÍXELES
    Total = 301 valores
    
    Parámetros:
        landmarks: lista de puntos MediaPipe (468 landmarks)
        frame_width: ancho frame webcam
        frame_height: alto frame webcam
        normalize: DEBE SER FALSE para coincidir con el dataset
    
    Return:
        vector numpy de shape (301,)
    """
    # Primer valor: timestamp simulado (valor promedio del dataset de entrenamiento)
    extracted = [1390385453.0]  
    
    for idx in DATASET_POINTS:
        if idx < len(landmarks):
            lm = landmarks[idx]
            
            # Convertir a coordenadas absolutas en píxeles (como en el dataset)
            x = lm.x * frame_width
            y = lm.y * frame_height
            z = lm.z * frame_width  # Escalar z también con frame_width
            
            extracted.extend([x, y, z])
        else:
            # Si faltan landmarks, rellenar con valores medios
            extracted.extend([frame_width/2, frame_height/2, 500.0])
    
    extracted = np.array(extracted, dtype=np.float32)
    
    # Verificar que tenga exactamente 301 valores
    assert len(extracted) == 301, f"Se esperaban 301 valores pero se generaron {len(extracted)}"
    
    return extracted
