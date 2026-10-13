# Q-MTL-XAI Dashboard

Dashboard interactivo para el framework Q-MTL-XAI de clasificación de ECG.

## Características

- **Predicción**: sube un archivo ECG (.npy, .csv, .npz) y obtén la clasificación
- **Explicabilidad**: análisis XAI (saliency, IG, t-SNE, attention)
- **Info**: métricas del modelo, arquitectura, datasets

## Demo

[Ver dashboard en vivo](https://q-mtl-xai-dashboard.streamlit.app)

## Backend

El dashboard se conecta al backend FastAPI en:

https://q-mtl-xai-backend.onrender.com

## Ejecución local

pip install -r requirements.txt
streamlit run app.py

## Tareas soportadas

- **latido**: clasificación de latido individual (250 muestras)
- **ventana**: clasificación de ventana (750 muestras)
- **isc**: isquemia (12 leads × 2500 muestras)

## Formato de entrada

- `.npy`: array numpy 1D (latido, ventana) o 2D (isquemia)
- `.csv`: una columna (1D) o múltiples columnas (12 leads)
- `.npz`: clave 'signal' o primera clave del archivo
