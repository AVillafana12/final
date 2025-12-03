import cv2
import mediapipe as mp
from ExtractLandmarks import extract_landmarks, DATASET_POINTS
from predict_expression import predict_expression, model
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.4,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

# Tipos de expresiones del dataset
EXPRESIONES = {
    'affirmative': 'Afirmación',
    'conditional': 'Condicional',
    'doubt_question': 'Pregunta/Duda',
    'emphasis': 'Énfasis',
    'negative': 'Negación',
    'relative': 'Relativo',
    'topics': 'Tópicos',
    'wh_question': 'Pregunta (Qué/Cómo)',
    'yn_question': 'Pregunta (Sí/No)'
}

# Guía de expresiones faciales
GUIA_EXPRESIONES = """
🎭 EXPRESIONES FACIALES GRAMATICALES (Libras):
   - Afirmación: Movimiento de cabeza hacia arriba/abajo
   - Pregunta Sí/No: Cejas levantadas, ojos abiertos
   - Pregunta Qué/Cómo: Cejas fruncidas
   - Negación: Movimiento de cabeza lateral
   - Énfasis: Expresión facial marcada
   - Duda: Cejas levantadas, boca ligeramente abierta
"""

# Variables para suavizado de predicciones
prediction_history = []
HISTORY_SIZE = 5  # Promediar últimas 5 predicciones

# Umbral ajustable (puede modificarse con teclas)
THRESHOLD = 0.47
SHOW_ALL_LANDMARKS = False  # Alternar con tecla 'L'
SHOW_GUIDE = False  # Mostrar guía de expresiones

print("="*80)
print("🎯 DETECTOR DE EXPRESIONES FACIALES GRAMATICALES - Libras")
print("="*80)
print("\n✨ NUEVAS CARACTERÍSTICAS:")
print("\n🎮 CONTROLES MEJORADOS:")
print("   [L]   - Alterna entre mostrar TODOS los landmarks vs solo los 100 que usa el modelo")
print("   [G]   - Muestra/oculta la guía de expresiones faciales")
print("   [+]   - Ajusta el umbral de sensibilidad (más difícil detectar PRESENTE)")
print("   [-]   - Ajusta el umbral de sensibilidad (más fácil detectar PRESENTE)")
print("   [ESC] - Salir")
print("\n📺 VISUALIZACIÓN MEJORADA:")
print("   Ventana principal muestra:")
print("      • 'EXPRESIÓN GRAMATICAL: PRESENTE/AUSENTE' (más claro)")
print("      • Sugerencias cuando no detecta expresión")
print("      • Guía de expresiones faciales en la parte inferior")
print("      • Solo los 100 landmarks del modelo (puntos verdes) o todos")
print("      • Barra de probabilidad más grande")
print("\n   Ventana limpia muestra:")
print("      • Solo la cámara sin overlay")
print("      • Texto grande del estado")
print("\n" + "="*80)
print(f"🎯 Umbral inicial: {THRESHOLD:.2f}")
print(GUIA_EXPRESIONES)
print("="*80)

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

            # ---- 1. Dibujar landmarks ----
            if SHOW_ALL_LANDMARKS:
                # Dibujar todos los landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    face_landmarks,
                    mp_face_mesh.FACEMESH_TESSELATION
                )
            else:
                # Dibujar solo los 100 landmarks que usa el modelo
                for idx in DATASET_POINTS:
                    if idx < len(face_landmarks.landmark):
                        lm = face_landmarks.landmark[idx]
                        x = int(lm.x * w)
                        y = int(lm.y * h)
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

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
            estado = "PRESENTE" if pred_smooth == 1 else "AUSENTE"
            text = f"EXPRESION GRAMATICAL: {estado}"
            color = (0, 255, 0) if pred_smooth == 1 else (0, 0, 255)

            cv2.putText(frame, text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 3)
            
            # Sugerencia de expresión si está ausente
            if pred_smooth == 0:
                cv2.putText(frame, "Prueba: Cejas arriba, ojos grandes, boca abierta", 
                           (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)
            
            # Mostrar probabilidades
            cv2.putText(frame, f"Prob PRESENTE: {prob_si:.3f}", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.putText(frame, f"Prob AUSENTE: {prob_no:.3f}", (20, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Mostrar umbral y modo
            landmarks_mode = "TODOS" if SHOW_ALL_LANDMARKS else "100 DEL MODELO"
            cv2.putText(frame, f"Umbral: {THRESHOLD:.2f} | Landmarks: {landmarks_mode}", 
                       (20, 165), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
            # Barra de probabilidad visual
            bar_width = int(prob_si * 500)
            cv2.rectangle(frame, (20, 180), (20 + bar_width, 210), (0, 255, 0), -1)
            cv2.rectangle(frame, (20, 180), (520, 210), (255, 255, 255), 2)
            
            # Línea de umbral
            threshold_x = int(20 + THRESHOLD * 500)
            cv2.line(frame, (threshold_x, 175), (threshold_x, 215), (0, 0, 255), 3)
            
            # Mostrar guía si está activada
            if SHOW_GUIDE:
                guide_y = h - 200
                cv2.rectangle(frame, (10, guide_y - 10), (w - 10, h - 10), (0, 0, 0), -1)
                cv2.rectangle(frame, (10, guide_y - 10), (w - 10, h - 10), (255, 255, 255), 2)
                
                guia_text = [
                    "EXPRESIONES GRAMATICALES (Libras):",
                    "Pregunta Si/No: Cejas arriba + ojos grandes",
                    "Pregunta Que/Como: Cejas fruncidas",
                    "Negacion: Cabeza lateral",
                    "Enfasis: Expresion marcada",
                ]
                for i, line in enumerate(guia_text):
                    cv2.putText(frame, line, (20, guide_y + i * 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Crear frame limpio solo con la cámara (sin mesh ni info)
            ret2, clean_frame = cap.read()
            if ret2:
                # Solo mostrar el estado de la expresión
                cv2.putText(clean_frame, text, (30, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 4)

    cv2.imshow("Libras Expression Detector - Calibrado", frame)
    if results.multi_face_landmarks:
    # Manejo de teclas
        key = cv2.waitKey(1) & 0xFF
    if key == 27:  # ESC
        break
    elif key == ord('+') or key == ord('='):
        THRESHOLD = min(0.95, THRESHOLD + 0.01)
        print(f"🎯 Umbral: {THRESHOLD:.2f} (más difícil detectar PRESENTE)")
    elif key == ord('-') or key == ord('_'):
        THRESHOLD = max(0.05, THRESHOLD - 0.01)
    cv2.imshow("Libras Expression Detector - Calibrado", frame)
    if results.multi_face_landmarks and ret2:
        cv2.imshow("Camara Limpia", clean_frame)
    
    # Manejo de teclasARKS = not SHOW_ALL_LANDMARKS
    elif key == ord('*') or key == ord('*'):
        SHOW_GUIDE = not SHOW_GUIDE
        print(f"📋 Guía: {'Visible' if SHOW_GUIDE else 'Oculta'}")
        THRESHOLD = max(0.05, THRESHOLD - 0.05)
        print(f"🎯 Umbral: {THRESHOLD:.2f} (más fácil detectar SI)")

cap.release()
cv2.destroyAllWindows()
print(f"\n✅ Umbral final: {THRESHOLD:.2f}")
