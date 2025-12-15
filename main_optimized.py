import cv2
import mediapipe as mp
from ExtractLandmarks import DATASET_POINTS
import numpy as np
import pickle

# Cargar modelo
with open('modelo_facial_expressions.pkl', 'rb') as f:
    model = pickle.load(f)

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

# Variables de calibración
baseline_vector = None
calibration_frames = []
CALIBRATION_COUNT = 30  # Frames para calibrar
is_calibrated = False

# Variables para suavizado
prediction_history = []
HISTORY_SIZE = 10
THRESHOLD = 0.57
SHOW_ALL_LANDMARKS = False
SHOW_GUIDE = True
show_threshold_bar = True
button_rect = (0, 0, 0, 0)  # (x, y, w, h) de botón toggle

def extract_landmarks_optimized(landmarks, frame_width, frame_height):
    """Extrae landmarks con normalización mejorada"""
    # Timestamp simulado
    extracted = [1390385453.0]
    
    for idx in DATASET_POINTS:
        if idx < len(landmarks):
            lm = landmarks[idx]
            x = lm.x * frame_width
            y = lm.y * frame_height
            z = lm.z * frame_width
            extracted.extend([x, y, z])
        else:
            extracted.extend([frame_width/2, frame_height/2, 500.0])
    
    return np.array(extracted, dtype=np.float32)

def calculate_relative_features(vector, baseline):
    """Calcula características relativas al estado neutral"""
    if baseline is None:
        return vector
    # Diferencia con respecto al baseline
    diff = vector - baseline
    return diff

def mouse_callback(event, x, y, flags, param):
    """Toggle barra de umbral al hacer clic en el botón."""
    global show_threshold_bar, button_rect
    if event == cv2.EVENT_LBUTTONDOWN:
        bx, by, bw, bh = button_rect
        if bx <= x <= bx + bw and by <= y <= by + bh:
            show_threshold_bar = not show_threshold_bar

print("="*80)
print("🎯 DETECTOR OPTIMIZADO - Expresiones Faciales Gramaticales")
print("="*80)
print("\n📋 INSTRUCCIONES:")
print("1. Mantén tu cara NEUTRAL durante los primeros segundos")
print("2. El sistema calibrará automáticamente")
print("3. Después haz expresiones faciales EXAGERADAS")
print("\n🎮 CONTROLES:")
print("   [R]   - Re-calibrar (cara neutral)")
print("   [L]   - Alternar vista de landmarks")
print("   [+/-] - Ajustar umbral")
print("   [ESC] - Salir")
print("\n💡 TIPS:")
print("   • Expresiones MUY marcadas: Cejas ARRIBA, ojos GRANDES, boca ABIERTA")
print("   • Movimientos de cabeza: Sí/No, lateral")
print("   • Prueba combinar: cejas + boca + ojos")
print("="*80)

cap = cv2.VideoCapture(0)
cv2.namedWindow("Detector Optimizado")
cv2.setMouseCallback("Detector Optimizado", mouse_callback)

# Ajustar resolución de webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 768)   # +20% ancho
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 576)  # +20% alto

calibration_counter = 0
expression_count = 0
previous_pred = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Invertir la cámara (vista espejo)
    frame = cv2.flip(frame, 1)

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    h, w, _ = frame.shape
    
    # Definir posición del botón (esquina superior derecha)
    btn_w, btn_h = 180, 40
    btn_x, btn_y = w - btn_w - 20, 20
    button_rect = (btn_x, btn_y, btn_w, btn_h)
    
    # Dibujar botón
    cv2.rectangle(frame, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), (255, 255, 255), -1)
    cv2.rectangle(frame, (btn_x, btn_y), (btn_x + btn_w, btn_y + btn_h), (0, 0, 0), 2)
    btn_label = "Ocultar umbral" if show_threshold_bar else "Mostrar umbral"
    (lbl_w, lbl_h), _ = cv2.getTextSize(btn_label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(frame, btn_label, (btn_x + (btn_w - lbl_w)//2, btn_y + btn_h//2 + lbl_h//2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

    # Cruz central para marcar el centro de la pantalla
    center_x, center_y = w // 2, h // 2
    cv2.line(frame, (center_x, 0), (center_x, h), (255, 255, 255), 1)
    cv2.line(frame, (0, center_y), (w, center_y), (255, 255, 255), 1)
    # Óvalo guía del tamaño aproximado de un rostro
    face_axis_x = int(w * 0.15)
    face_axis_y = int(h * 0.22)
    cv2.ellipse(frame, (center_x, center_y), (face_axis_x, face_axis_y), 0, 0, 360, (255, 255, 255), 2)

    results = face_mesh.process(frame_rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            # Dibujar landmarks
            if SHOW_ALL_LANDMARKS:
                mp_drawing.draw_landmarks(frame, face_landmarks, mp_face_mesh.FACEMESH_TESSELATION)
            else:
                for idx in DATASET_POINTS:
                    if idx < len(face_landmarks.landmark):
                        lm = face_landmarks.landmark[idx]
                        x, y = int(lm.x * w), int(lm.y * h)
                        cv2.circle(frame, (x, y), 3, (0, 255, 0), -1)

            # Extraer vector
            vector = extract_landmarks_optimized(face_landmarks.landmark, w, h)

            # FASE DE CALIBRACIÓN
            if not is_calibrated:
                calibration_frames.append(vector)
                calibration_counter += 1
                
                # Barra de calibración
                progress = int((calibration_counter / CALIBRATION_COUNT) * 500)
                cv2.rectangle(frame, (70, 200), (70 + progress, 230), (0, 255, 255), -1)
                cv2.rectangle(frame, (70, 200), (570, 230), (255, 255, 255), 2)
                
                cv2.putText(frame, "CALIBRANDO... Mantén cara NEUTRAL", (70, 180),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                cv2.putText(frame, f"{calibration_counter}/{CALIBRATION_COUNT}", (250, 220),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if calibration_counter >= CALIBRATION_COUNT:
                    baseline_vector = np.mean(calibration_frames, axis=0)
                    is_calibrated = True
                    print("✅ Calibración completada! Ahora haz expresiones exageradas")
            
            # FASE DE DETECCIÓN
            else:
                # Calcular características relativas
                relative_features = calculate_relative_features(vector, baseline_vector)
                
                # Usar magnitud del cambio como factor
                change_magnitude = np.linalg.norm(relative_features) / 1000.0
                
                # Predicción con el modelo
                vector_reshaped = vector.reshape(1, -1)
                proba = model.predict_proba(vector_reshaped)[0]
                prob_no, prob_si = proba[0], proba[1]
                
                # Ajustar probabilidad con magnitud del cambio
                prob_si_adjusted = min(1.0, prob_si + (change_magnitude * 0.3))
                
                pred = 1 if prob_si_adjusted > THRESHOLD else 0
                prediction_history.append(pred)
                if len(prediction_history) > HISTORY_SIZE:
                    prediction_history.pop(0)
                pred_smooth = 1 if np.mean(prediction_history) > 0.6 else 0
                
                # Contador de expresiones detectadas (transición 0 -> 1)
                if pred_smooth == 1 and previous_pred == 0:
                    expression_count += 1
                previous_pred = pred_smooth

                # Visualización
                estado = "PRESENTE" if pred_smooth == 1 else "AUSENTE"
                color = (0, 255, 0) if pred_smooth == 1 else (0, 0, 255)
                expr_text = f"EXPRESION: {estado}"
                expr_scale = 1.44  # 20% más grande que 1.2
                if not show_threshold_bar:
                    (expr_w, expr_h), _ = cv2.getTextSize(expr_text, cv2.FONT_HERSHEY_SIMPLEX, expr_scale, 3)
                    expr_x = (w - expr_w) // 2
                    expr_y = 60
                    # Fondo negro para legibilidad
                    padding = 10
                    cv2.rectangle(frame, (expr_x - padding, expr_y - expr_h - padding),
                                  (expr_x + expr_w + padding, expr_y + padding),
                                  (0, 0, 0), -1)
                    cv2.putText(frame, expr_text, (expr_x, expr_y),
                                cv2.FONT_HERSHEY_SIMPLEX, expr_scale, color, 3)
                if pred_smooth == 0:
                    cv2.putText(frame, "HAZ EXPRESION EXAGERADA!", (20, 90),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
                
                
                if show_threshold_bar:
                    cv2.putText(frame, f"Umbral: {THRESHOLD:.2f}", (20, 160),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                    
                    # Barra de probabilidad
                    bar_width = int(prob_si_adjusted * 500)
                    cv2.rectangle(frame, (20, 180), (20 + bar_width, 210), color, -1)
                    cv2.rectangle(frame, (20, 180), (520, 210), (255, 255, 255), 2)
                    threshold_x = int(20 + THRESHOLD * 500)
                    cv2.line(frame, (threshold_x, 175), (threshold_x, 215), (0, 0, 255), 3)
                
                # Guía
                if SHOW_GUIDE:
                    guide_y = h - 150
                    cv2.rectangle(frame, (10, guide_y), (w-10, h-10), (0, 0, 0), -1)
                    cv2.rectangle(frame, (10, guide_y), (w-10, h-10), (255, 255, 255), 2)
                    tips = [
                        "TIPS: Expresiones MUY EXAGERADAS",
                        "• Cejas ARRIBA + Ojos MUY abiertos",
                        "• Boca BIEN abierta o sonrisa grande",
                        "• Movimiento de cabeza marcado"
                    ]
                    for i, tip in enumerate(tips):
                        cv2.putText(frame, tip, (20, guide_y + 25 + i*25),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imshow("Detector Optimizado", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break
    elif key == ord('r') or key == ord('R'):
        is_calibrated = False
        calibration_frames = []
        calibration_counter = 0
        prediction_history = []
        print("🔄 Re-calibrando... Mantén cara neutral")
    elif key == ord('+') or key == ord('='):
        THRESHOLD = min(0.95, THRESHOLD + 0.05)
        print(f"🎯 Umbral: {THRESHOLD:.2f}")
    elif key == ord('-'):
        THRESHOLD = max(0.05, THRESHOLD - 0.05)
        print(f"🎯 Umbral: {THRESHOLD:.2f}")
    elif key == ord('l') or key == ord('L'):
        SHOW_ALL_LANDMARKS = not SHOW_ALL_LANDMARKS
    elif key == ord('g') or key == ord('G'):
        SHOW_GUIDE = not SHOW_GUIDE

cap.release()
cv2.destroyAllWindows()
print("\n✅ Detector cerrado")
