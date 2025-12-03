import cv2
import mediapipe as mp
from ExtractLandmarks import extract_landmarks
from predict_expression import predict_expression

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

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

            # ---- 2. Extraer 100 puntos → vector 301 ----
            # IMPORTANTE: normalize=False para usar valores absolutos como el dataset
            vector = extract_landmarks(
                landmarks=face_landmarks.landmark,
                frame_width=w,
                frame_height=h,
                normalize=False
            )

            # ---- 3. Predicción del modelo ----
            pred = predict_expression(vector)

            text = "EXPRESIÓN: SI" if pred == 1 else "EXPRESIÓN: NO"
            
            # Color según predicción
            color = (0, 255, 0) if pred == 1 else (0, 0, 255)  # Verde para SI, Rojo para NO

            # ---- 4. Mostrar resultado ----
            cv2.putText(frame, text, (30, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, color, 3)
            
            # Mostrar probabilidad si es Random Forest
            cv2.putText(frame, f"Prediccion: {pred}", (30, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow("Libras Grammatical Expression Detector", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC
        break

cap.release()
cv2.destroyAllWindows()
