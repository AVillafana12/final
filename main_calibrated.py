import cv2
import mediapipe as mp
from ExtractLandmarks import extract_landmarks
from predict_expression import predict_expression, model
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

# Variables para suavizado de predicciones
prediction_history = []
HISTORY_SIZE = 5  # Promediar últimas 5 predicciones

# Umbral ajustable (puede modificarse con teclas)
THRESHOLD = 0.5
print("🎮 CONTROLES:")
print("   [+] Aumentar umbral (más difícil detectar SI)")
print("   [-] Disminuir umbral (más fácil detectar SI)")
print("   [ESC] Salir")
print(f"\n🎯 Umbral inicial: {THRESHOLD:.2f}\n")

# Iniciar webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape

    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # ---- 1. Dibujar Face Mesh ----
            mp_drawing.draw_landmarks(
                frame,
                face_landmarks,
                mp_face_mesh.FACEMESH_TESSELATION
            )

            # ---- 2. Extraer landmarks ----
            vector = extract_landmarks(
                landmarks=face_landmarks.landmark,
                frame_width=w,
                frame_height=h,
                normalize=False
            )

            # ---- 3. Obtener probabilidad ----
            vector_reshaped = np.array(vector).reshape(1, -1)
            proba = model.predict_proba(vector_reshaped)[0]
            prob_no = proba[0]   # Probabilidad de clase 0 (NO)
            prob_si = proba[1]   # Probabilidad de clase 1 (SI)
            
            # ---- 4. Predicción con umbral ajustable ----
            pred = 1 if prob_si > THRESHOLD else 0
            
            # Suavizado: usar promedio de últimas predicciones
            prediction_history.append(pred)
            if len(prediction_history) > HISTORY_SIZE:
                prediction_history.pop(0)
            pred_smooth = 1 if np.mean(prediction_history) > 0.5 else 0

            # ---- 5. Visualización ----
            text = "EXPRESIÓN: SI" if pred_smooth == 1 else "EXPRESIÓN: NO"
            color = (0, 255, 0) if pred_smooth == 1 else (0, 0, 255)

            cv2.putText(frame, text, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
            
            # Mostrar probabilidades
            cv2.putText(frame, f"Prob SI: {prob_si:.3f}", (30, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, f"Prob NO: {prob_no:.3f}", (30, 130),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Mostrar umbral actual
            cv2.putText(frame, f"Umbral: {THRESHOLD:.2f}", (30, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            # Barra de probabilidad visual
            bar_width = int(prob_si * 400)
            cv2.rectangle(frame, (30, 180), (30 + bar_width, 210), (0, 255, 0), -1)
            cv2.rectangle(frame, (30, 180), (430, 210), (255, 255, 255), 2)
            
            # Línea de umbral
            threshold_x = int(30 + THRESHOLD * 400)
            cv2.line(frame, (threshold_x, 175), (threshold_x, 215), (0, 0, 255), 3)
            
            # Crear frame limpio solo con la cámara (sin mesh)
            clean_frame = frame.copy()
            # Agregar solo el texto de la predicción
            cv2.putText(clean_frame, text, (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)

    cv2.imshow("Libras Expression Detector - Calibrado", frame)
    if results.multi_face_landmarks:
        cv2.imshow("Camara Limpia", clean_frame)

    # Manejo de teclas
    key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('+') or key == ord('='):
        THRESHOLD = min(0.95, THRESHOLD + 0.05)
        print(f"🎯 Umbral: {THRESHOLD:.2f} (más difícil detectar SI)")
    elif key == ord('-') or key == ord('_'):
        THRESHOLD = max(0.05, THRESHOLD - 0.05)
        print(f"🎯 Umbral: {THRESHOLD:.2f} (más fácil detectar SI)")

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Umbral final: {THRESHOLD:.2f}")
