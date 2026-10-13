# app.py
# Dashboard Q-MTL-XAI con Streamlit
# 3 tabs: Predicción, Explicabilidad (XAI), Info

import os
import json
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from PIL import Image

# === Configuración ===
BACKEND_URL = os.environ.get(
    "BACKEND_URL",
    "https://q-mtl-xai-backend.onrender.com"
)

ASSETS_DIR = os.path.join(os.path.dirname(__file__), "assets")
XAI_DIR = os.path.join(ASSETS_DIR, "xai")

CLASS_NAMES = {
    'latido': ['normal', 'paced', 'PVC', 'SPB'],
    'ventana': ['normal', 'paced', 'PVC', 'SPB'],
    'isc': ['normal', 'ischemia'],
}

# === Configuración de la página ===
st.set_page_config(
    page_title="Q-MTL-XAI Dashboard",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# === CSS personalizado ===
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 1rem;
        text-align: center;
        margin: 1rem 0;
    }
    .prediction-label {
        font-size: 3rem;
        font-weight: bold;
    }
    .confidence-label {
        font-size: 1.2rem;
        opacity: 0.9;
    }
    .sqi-good {
        color: #28a745;
        font-weight: bold;
    }
    .sqi-fair {
        color: #ffc107;
        font-weight: bold;
    }
    .sqi-poor {
        color: #dc3545;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


# === Funciones auxiliares ===

def load_uploaded_file(uploaded_file):
    """Carga un archivo subido (.npy, .csv, .npz)."""
    if uploaded_file.name.endswith('.npy'):
        signal = np.load(uploaded_file, allow_pickle=True)
    elif uploaded_file.name.endswith('.npz'):
        data = np.load(uploaded_file, allow_pickle=True)
        # Buscar la key 'signal' o tomar la primera
        if 'signal' in data:
            signal = data['signal']
        else:
            signal = data[list(data.keys())[0]]
    elif uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
        # Si tiene 1 columna, es 1D
        if df.shape[1] == 1:
            signal = df.iloc[:, 0].values
        else:
            # Multi-canal: cada columna es un lead
            signal = df.values.T  # (n_leads, n_samples)
    else:
        raise ValueError(f"Formato no soportado: {uploaded_file.name}")

    signal = np.asarray(signal, dtype=np.float32)
    return signal


def call_backend(signal, task, sampling_rate=250):
    """Llama al backend /predict."""
    payload = {
        "signal": signal.flatten().tolist(),
        "task": task,
        "sampling_rate": sampling_rate,
    }

    r = requests.post(f"{BACKEND_URL}/predict", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()


def plot_signal(signal, title="Señal ECG"):
    """Grafica la señal con Plotly."""
    if signal.ndim == 1:
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=signal,
            mode='lines',
            line=dict(color='#1f77b4', width=1.5),
            name='ECG',
        ))
    else:
        fig = go.Figure()
        for i in range(signal.shape[0]):
            fig.add_trace(go.Scatter(
                y=signal[i],
                mode='lines',
                line=dict(width=1),
                name=f'Lead {i+1}',
                visible=(i == 0),
            ))
        fig.update_layout(
            updatemenus=[dict(
                type="buttons",
                buttons=[
                    dict(label=f"Lead {i+1}",
                         method="update",
                         args=[{"visible": [j == i for j in range(signal.shape[0])]}])
                    for i in range(signal.shape[0])
                ],
            )]
        )

    fig.update_layout(
        title=title,
        xaxis_title="Tiempo (muestras)",
        yaxis_title="Amplitud",
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white",
    )
    return fig


def plot_probabilities(probs_dict):
    """Gráfico de barras de probabilidades."""
    classes = list(probs_dict.keys())
    values = list(probs_dict.values())

    fig = go.Figure(data=[
        go.Bar(
            x=classes,
            y=values,
            marker_color=['#1f77b4' if v == max(values) else '#aec7e8' for v in values],
            text=[f'{v:.3f}' for v in values],
            textposition='auto',
        )
    ])
    fig.update_layout(
        title="Probabilidades por clase",
        yaxis_title="Probabilidad",
        yaxis_range=[0, 1],
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        template="plotly_white",
    )
    return fig


def get_sqi_color(sqi):
    return {'Good': 'sqi-good', 'Fair': 'sqi-fair', 'Poor': 'sqi-poor'}.get(sqi, '')


# === Header ===
st.markdown('<div class="main-title">🫀 Q-MTL-XAI Dashboard</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Clasificación multi-task de ECG: arritmias, isquemia y calidad de señal</div>',
    unsafe_allow_html=True
)

# === Sidebar ===
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/heart-with-pulse.png", width=80)
    st.title("Configuración")

    st.markdown("### Backend")
    st.code(BACKEND_URL, language="text")

    # Verificar estado
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=10)
        if r.status_code == 200:
            st.success("🟢 Backend online")
        else:
            st.warning(f"🟡 Backend respondió {r.status_code}")
    except Exception:
        st.info("🔵 Backend dormido (tarda al despertar)")

    st.markdown("---")
    st.markdown("### Sobre el proyecto")
    st.markdown("""
    **Q-MTL-XAI** es un framework de aprendizaje multi-task para:
    - **Arritmias** (4 clases)
    - **Isquemia** (2 clases)
    - **Calidad de señal** (SQI)
    
    Datasets: MIT-BIH, PTB-XL, WF-PPG, NSTDB.
    """)

    st.markdown("---")
    st.caption("Modelo: MultiTaskEncoder (725,932 params)")


# === Tabs ===
tab1, tab2, tab3 = st.tabs(["🔍 Predicción", "🧠 Explicabilidad", "ℹ️ Info del modelo"])


# === TAB 1: Predicción ===
with tab1:
    st.markdown("### Sube una señal ECG")

    col1, col2 = st.columns([2, 1])

    with col1:
        task = st.selectbox(
            "Tarea de clasificación",
            options=['latido', 'ventana', 'isc'],
            format_func=lambda x: {
                'latido': 'Latido (250 muestras)',
                'ventana': 'Ventana (750 muestras)',
                'isc': 'Isquemia (12 leads × 2500 muestras)',
            }[x],
        )

    with col2:
        sampling_rate = st.number_input(
            "Sampling rate (Hz)",
            min_value=50,
            max_value=1000,
            value=250,
            step=10,
        )

    uploaded_file = st.file_uploader(
        "Sube un archivo (.npy, .csv, .npz)",
        type=['npy', 'csv', 'npz'],
        help="1D para latido/ventana, 2D para isquemia (12 leads)",
    )

    if uploaded_file is not None:
        try:
            signal = load_uploaded_file(uploaded_file)

            st.success(f"✅ Archivo cargado: shape={signal.shape}")

            # Mostrar señal
            fig = plot_signal(signal, title="Señal cargada")
            st.plotly_chart(fig, use_container_width=True)

            # Botón predecir
            if st.button("🔮 Predecir", type="primary", use_container_width=True):
                with st.spinner("Enviando al backend..."):
                    try:
                        result = call_backend(signal, task, sampling_rate)

                        # Mostrar resultado
                        st.markdown("---")
                        st.markdown("### Resultado")

                        # Predicción principal
                        sqi_class = get_sqi_color(result['sqi'])
                        st.markdown(f"""
                        <div class="prediction-box">
                            <div class="prediction-label">{result['prediction'].upper()}</div>
                            <div class="confidence-label">Confianza: {result['confidence']:.2%}</div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Métricas
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Predicción", result['prediction'])
                        m2.metric("Confianza", f"{result['confidence']:.2%}")
                        m3.metric("SQI", result['sqi'])

                        # Probabilidades
                        fig_probs = plot_probabilities(result['probabilities'])
                        st.plotly_chart(fig_probs, use_container_width=True)

                    except Exception as e:
                        st.error(f"Error al llamar al backend: {e}")

        except Exception as e:
            st.error(f"Error al cargar el archivo: {e}")

    else:
        st.info("👆 Sube un archivo para empezar. También puedes explorar las pestañas de Explicabilidad e Info.")


# === TAB 2: Explicabilidad ===
with tab2:
    st.markdown("### 🧠 Explicabilidad del modelo")
    st.markdown("Análisis XAI pre-generado sobre el modelo unificado.")

    # Verificar si hay imágenes XAI
    if os.path.exists(XAI_DIR):
        xai_files = sorted([f for f in os.listdir(XAI_DIR) if f.endswith('.png')])

        if xai_files:
            st.markdown(f"**{len(xai_files)} análisis disponibles**")

            # Agrupar por tipo
            categories = {
                'Saliency Maps': [f for f in xai_files if 'saliency' in f],
                'Integrated Gradients': [f for f in xai_files if 'ig_' in f],
                't-SNE de Embeddings': [f for f in xai_files if 'tsne' in f],
                'Attention Weights': [f for f in xai_files if 'attention' in f],
                'Otros análisis': [f for f in xai_files if not any(x in f for x in ['saliency', 'ig_', 'tsne', 'attention'])],
            }

            for cat_name, cat_files in categories.items():
                if cat_files:
                    st.markdown(f"#### {cat_name}")
                    for img_file in cat_files:
                        img_path = os.path.join(XAI_DIR, img_file)
                        with st.expander(f"📊 {img_file}"):
                            st.image(img_path, use_column_width=True)
        else:
            st.warning("No hay imágenes XAI disponibles.")
    else:
        st.warning(f"No se encontró la carpeta {XAI_DIR}")


# === TAB 3: Info ===
with tab3:
    st.markdown("### ℹ️ Información del modelo")

    # Métricas
    metrics_path = os.path.join(ASSETS_DIR, "encoder_metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

        st.markdown("#### Métricas del modelo (test)")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**Head 1: Latido**")
            st.metric("F1 macro", f"{metrics['head_1_latido']['f1_macro']:.4f}")
            st.metric("Accuracy", f"{metrics['head_1_latido']['accuracy']:.4f}")

        with col2:
            st.markdown("**Head 2: Ventana**")
            st.metric("F1 macro", f"{metrics['head_2_ventana']['f1_macro']:.4f}")
            st.metric("Accuracy", f"{metrics['head_2_ventana']['accuracy']:.4f}")

        with col3:
            st.markdown("**Head 3: Isquemia**")
            st.metric("F1 macro", f"{metrics['head_3_isquemia']['f1_macro']:.4f}")
            st.metric("Accuracy", f"{metrics['head_3_isquemia']['accuracy']:.4f}")

    st.markdown("---")
    st.markdown("#### Arquitectura")
    st.markdown("""
    - **Encoder compartido**: CNN 1D (32, 64, 128 filtros) + LSTM bidireccional + Attention Pooling
    - **Head 1 (latido)**: 4 clases (normal, paced, PVC, SPB)
    - **Head 2 (ventana)**: 4 clases (normal, paced, PVC, SPB)
    - **Head 3 (isquemia)**: 2 clases (normal, ischemia) + encoder 12 leads separado
    - **Parámetros totales**: 725,932
    """)

    st.markdown("---")
    st.markdown("#### Datasets")
    st.markdown("""
    - **MIT-BIH**: arritmias (48 registros, ~112k latidos)
    - **PTB-XL**: isquemia (17,755 registros, 5,000 usados)
    - **WF-PPG**: calidad de señal (27 sujetos × 6 sesiones)
    - **NSTDB**: ruido (15 registros)
    """)

    st.markdown("---")
    st.caption("Q-MTL-XAI © 2026 — Framework de clasificación ECG con XAI")


# === Footer ===
st.markdown("---")
st.caption(f"Backend: [{BACKEND_URL}]({BACKEND_URL}) | Modelo: MultiTaskEncoder | v1.0.0")
