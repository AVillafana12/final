# 🎭 Detector de Expresiones Faciales Gramaticales (Libras)

Sistema de reconocimiento en tiempo real de expresiones faciales gramaticales utilizadas en la Lengua de Señas Brasileña (Libras), basado en Machine Learning y MediaPipe Face Mesh.

## 📋 Descripción del Proyecto

Este proyecto implementa un clasificador binario de expresiones faciales que identifica gestos gramaticales no manuales utilizados en Libras. El sistema captura puntos faciales (landmarks) mediante webcam y predice la expresión en tiempo real usando un modelo de Machine Learning entrenado.

###  Características Principales

- **Detección facial en tiempo real** con MediaPipe Face Mesh (468 landmarks)
- **Clasificación binaria** de expresiones faciales gramaticales
- **Sistema de calibración** para mejorar la precisión
- **Suavizado de predicciones** mediante historial temporal
- **Múltiples implementaciones** optimizadas para diferentes casos de uso
- **Interfaz visual interactiva** con controles por teclado

##  Arquitectura del Sistema

### Componentes Principales

```
├── final.ipynb                          # Notebook de entrenamiento y análisis EDA
├── grammatical_facial_expressions.csv   # Dataset de entrenamiento
├── modelo_facial_expressions.pkl        # Modelo entrenado
├── scaler_facial_expressions.pkl        # Scaler para normalización
├── modelo_info.pkl                      # Metadata del modelo
├── ExtractLandmarks.py                  # Extracción de 100 landmarks clave
├── predict_expression.py                # Motor de predicción
├── FaceCapture.py                       # Captura básica de rostro
├── main.py                              # Sistema básico de detección
├── main_calibrated.py                   # Sistema con calibración
├── main_optimized.py                    # Sistema optimizado
└── test_prediction.py                   # Script de pruebas
```

### Pipeline de Procesamiento

```
Webcam → MediaPipe Face Mesh → Extracción de 100 Landmarks 
→ Vector de 301 características → Normalización (opcional) 
→ Modelo ML → Predicción (0 o 1) → Visualización
```

## 🔬 Metodología de Machine Learning

### Dataset

- **Formato**: 301 características por muestra
  - 1 timestamp
  - 100 landmarks × 3 coordenadas (x, y, z) = 300 valores
- **Tipo**: Clasificación binaria (0/1)
- **Preprocesamiento**: Valores en píxeles absolutos

### Modelos Evaluados

El notebook `final.ipynb` entrena y compara múltiples algoritmos:

-  **Random Forest** (mejor rendimiento)
- Support Vector Machine (Linear y RBF)
- K-Nearest Neighbors
- Naive Bayes
- Decision Tree
- Gradient Boosting
- AdaBoost
- Neural Network (MLP)
- Linear Discriminant Analysis (LDA)

### Métricas de Evaluación

- Accuracy
- Precision
- Recall
- F1-Score
- Matriz de Confusión
- Tiempo de entrenamiento

### Análisis Exploratorio (EDA)

- Visualización con LDA para separación de clases
- Profiling completo del dataset con `ydata-profiling`
- Análisis de distribución de características

##  Instalación

### Requisitos Previos

- Python 3.11+
- Webcam funcional
- Windows/Linux/MacOS

### Paso 1: Clonar el Repositorio

```bash
git clone https://github.com/AVillafana12/final.git
cd final
```

### Paso 2: Crear Entorno Virtual

```bash
# Windows PowerShell
python -m venv env
.\env\Scripts\Activate.ps1

# Linux/Mac
python3 -m venv env
source env/bin/activate
```

### Paso 3: Instalar Dependencias

```bash
pip install -r requirements.txt
```

### Dependencias Principales

```
opencv-python==4.10.0.84
mediapipe==0.10.14
scikit-learn==1.6.1
pandas==2.2.0
numpy==2.2.6
matplotlib==3.10.0
seaborn==0.13.2
ydata-profiling==4.13.1
```

## 💻 Uso del Sistema

### Opción 1: Sistema Básico

```bash
python main.py
```

**Características**:
- Detección y predicción en tiempo real
- Visualización de Face Mesh completo
- Sin calibración

### Opción 2: Sistema Calibrado (Recomendado)

```bash
python main_calibrated.py
```

**Características**:
- Sistema de calibración inicial (30 frames en posición neutral)
- Suavizado de predicciones con historial
- Umbral ajustable en tiempo real
- Controles interactivos avanzados

**Controles**:
- `ESPACIO`: Iniciar/reiniciar calibración
- `L`: Alternar visualización de todos los landmarks
- `G`: Mostrar/ocultar guía de expresiones
- `+/-`: Ajustar umbral de detección
- `Q`: Salir

### Opción 3: Sistema Optimizado

```bash
python main_optimized.py
```

**Características**:
- Extracción optimizada de landmarks
- Mayor rendimiento
- Calibración automática

### Script de Pruebas

Verificar el funcionamiento del modelo con datos del dataset:

```bash
python test_prediction.py
```

##  Entrenamiento del Modelo

### Ejecutar Notebook de Entrenamiento

1. Abrir `final.ipynb` en Jupyter Notebook o VS Code
2. Ejecutar todas las celdas secuencialmente
3. El notebook realizará:
   - Carga y análisis del dataset
   - EDA con visualizaciones
   - Entrenamiento de múltiples modelos
   - Comparación de rendimiento
   - Selección automática del mejor modelo
   - Guardado de archivos `.pkl`

### Archivos Generados

- `modelo_facial_expressions.pkl`: Modelo entrenado
- `scaler_facial_expressions.pkl`: Scaler para normalización
- `modelo_info.pkl`: Metadata (nombre, métricas, configuración)

## 🎯 Expresiones Detectadas

El sistema está diseñado para detectar expresiones faciales gramaticales de Libras:

- **Afirmación**: Movimiento de cabeza hacia arriba/abajo
- **Pregunta Sí/No**: Cejas levantadas, ojos abiertos
- **Pregunta Qué/Cómo**: Cejas fruncidas
- **Negación**: Movimiento de cabeza lateral
- **Énfasis**: Expresión facial marcada
- **Duda**: Cejas levantadas, boca ligeramente abierta
- **Condicional**
- **Relativo**
- **Tópicos**

##  Configuración Técnica

### MediaPipe Face Mesh

```python
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,              # Una cara a la vez
    refine_landmarks=True,        # Incluir iris (478 landmarks)
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
```

### Landmarks Extraídos

El sistema extrae **100 puntos clave** de los 478 disponibles:

- 17 puntos del contorno facial
- 10 puntos de cejas (5 por ceja)
- 16 puntos de ojos
- 9 puntos de nariz
- 20 puntos de boca (exterior e interior)
- 10 puntos de iris
- 18 puntos adicionales estratégicos

### Vector de Características

- **Tamaño**: 301 valores
  - Índice 0: Timestamp/Frame ID
  - Índices 1-300: Coordenadas (x, y, z) de 100 landmarks
- **Formato**: Valores en píxeles (no normalizados por defecto)

##  Rendimiento

El modelo seleccionado automáticamente (típicamente Random Forest) alcanza:

- **Accuracy**: ~95%+ (depende del dataset)
- **F1-Score**: ~95%+
- **Tiempo de predicción**: <10ms por frame
- **FPS**: 25-30 en tiempo real

##  Personalización

### Ajustar Umbral de Detección

En `main_calibrated.py`:

```python
THRESHOLD = 0.47  # Ajustar entre 0.0 y 1.0
```

### Cambiar Tamaño de Historial de Suavizado

```python
HISTORY_SIZE = 10  # Número de predicciones a promediar
```

### Modificar Parámetros de Calibración

```python
CALIBRATION_COUNT = 30  # Frames para calibración inicial
```

##  Estructura de Datos

### Formato del Dataset CSV

```
0,expression,target
"1390385453.0 x1 y1 z1 x2 y2 z2 ...",affirmative,0
"1390385454.0 x1 y1 z1 x2 y2 z2 ...",yn_question,1
```

##  Troubleshooting

### Error: Webcam no detectada

```bash
# Verificar disponibilidad de cámara
python -c "import cv2; print(cv2.VideoCapture(0).isOpened())"
```

### Error: Modelo no encontrado

Asegúrate de ejecutar el notebook `final.ipynb` completo para generar los archivos `.pkl`.

### Baja precisión en detección

1. Mejorar iluminación
2. Ajustar el umbral con teclas `+/-`
3. Recalibrar con `ESPACIO`
4. Verificar que el rostro esté centrado y visible

### Predicciones inestables

Aumentar `HISTORY_SIZE` para mayor suavizado (menor reactividad).

##  Contribuciones

Contribuciones son bienvenidas. Por favor:

1. Fork el repositorio
2. Crea una rama para tu feature (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add some AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

##  Licencia

Este proyecto es de código abierto y está disponible bajo la licencia especificada en el repositorio.

##  Autores

- **Alex Villafana** - [@AVillafana12](https://github.com/AVillafana12)


## 📚 Referencias

- [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [Libras - Lengua de Señas Brasileña](https://es.wikipedia.org/wiki/Lengua_de_se%C3%B1as_brasile%C3%B1a)
- [Scikit-learn Documentation](https://scikit-learn.org/)

##  Contacto

Para preguntas, sugerencias o problemas, por favor abre un issue en el repositorio de GitHub.

---

**Nota**: Este proyecto fue desarrollado como parte de un trabajo final de Machine Learning, enfocado en el reconocimiento de expresiones faciales gramaticales en Lengua de Señas.
