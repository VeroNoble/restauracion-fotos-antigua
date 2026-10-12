# Restauración de fotos antiguas con Deep Learning (GAN / U-Net)

Trabajo práctico de la Tecnicatura Superior en Ciencia de Datos e IA.

## Objetivo

Entrenar un modelo de deep learning capaz de eliminar el daño físico típico de fotos antiguas (grietas y manchas de humedad) en fotografías a color, en blanco y negro o en sepia, a partir de pares de imágenes dañadas/originales generados de forma sintética.

## Datos

Dataset [CelebFaces Dataset - 64x64](https://www.kaggle.com/datasets/arnrob/celeba-small-images-dataset) (Kaggle): ~50.000 imágenes de rostros de celebridades, recortadas y redimensionadas a 64x64 píxeles. Sobre estas imágenes originales se aplica un pipeline propio de "envejecimiento" sintético: viraje aleatorio de color (color / blanco y negro / sepia), grietas y manchas de humedad generadas proceduralmente. Se genera así un dataset pareado (foto dañada → foto original) con partición 80/20 en train/test.

## Enfoque

1. **Generación del dataset de daño**: funciones propias para aplicar viraje de color y daño físico simulado (grietas, manchas) sobre las imágenes originales.
2. **Arquitectura**: un generador tipo U-Net (encoder-decoder con conexiones salteadas) y un discriminador convolucional, entrenados en conjunto al estilo Pix2Pix (pérdida adversarial + pérdida L1).
3. **Entrenamiento**: bucle de entrenamiento custom con `tf.GradientTape`, optimizadores Adam para generador y discriminador, y visualización de resultados al final de cada época.
4. **Evaluación**: test de generalización sobre imágenes del conjunto de test no vistas durante el entrenamiento.

## Resultados

El modelo entrenado logra atenuar grietas y manchas en fotos con daño simulado, conservando el tono original de la imagen (color, blanco y negro o sepia). 
El repositorio incluye el modelo ya entrenado (`modelo_entrenado.keras`) para poder probarlo directamente sin necesidad de reentrenar.

## Herramientas

Python · TensorFlow / Keras · OpenCV · NumPy · Matplotlib

## Cómo correrlo

**Opción 1 — usar el modelo ya entrenado (recomendado):**

```python
import tensorflow as tf
generador = tf.keras.models.load_model('modelo_entrenado.keras')
# generador(imagen_danada_64x64x3, training=False) -> imagen restaurada
```

**Opción 2 — reproducir el entrenamiento completo:**

Este notebook fue desarrollado en Google Colab y usa rutas de Google Drive personales para el dataset original. Para reproducirlo hay que:

```bash
pip install -r requirements.txt
```

descargar el dataset de Kaggle y ajustar la variable `CARPETA_PROYECTO` (al inicio del notebook) a la carpeta de tu Google Drive o entorno local donde esté el ZIP. El entrenamiento completo (30 épocas) requiere GPU.
