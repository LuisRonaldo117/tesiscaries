import os
import tempfile
from pathlib import Path

import numpy as np
import requests
import streamlit as st
from dotenv import load_dotenv
from PIL import Image
import supervision as sv

load_dotenv()

ROBOFLOW_API_KEY = os.getenv("ROBOFLOW_API_KEY")
MODEL_ID = "dental-caries-i8vaj"
CONFIDENCE_THRESHOLD = 0.30

st.set_page_config(page_title="Detector de Caries Dental", layout="wide")


def validate_api_key() -> bool:
    if not ROBOFLOW_API_KEY:
        st.error(
            "Configura tu API Key de Roboflow como variable de entorno "
            "`ROBOFLOW_API_KEY` en los Secrets de Streamlit Cloud."
        )
        return False
    return True


def load_image(uploaded_file) -> Image.Image | None:
    try:
        return Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Error al abrir la imagen: {e}")
        return None


def run_inference(image_path: str, model_version: str) -> dict | None:
    model_id = f"{MODEL_ID}/{model_version}"
    url = f"https://detect.roboflow.com/{model_id}?api_key={ROBOFLOW_API_KEY}"
    try:
        with st.spinner("Analizando radiografía..."):
            with open(image_path, "rb") as f:
                resp = requests.post(url, files={"file": f})
            resp.raise_for_status()
            return resp.json()
    except Exception as e:
        st.error(f"Error en la inferencia: {e}")
        return None


def get_class_names(result: dict) -> dict[int, str]:
    names = {}
    for pred in result.get("predictions", []):
        cid = pred.get("class_id")
        cname = pred.get("class")
        if cid is not None and cname is not None:
            names[cid] = cname
    return names


def filter_by_confidence(
    detections: sv.Detections, threshold: float
) -> sv.Detections:
    if len(detections) == 0:
        return detections
    mask = detections.confidence >= threshold
    return detections[mask]


def build_labels(detections: sv.Detections, class_names: dict) -> list[str]:
    return ["caries detectada"] * len(detections)


def annotate_image(
    image: np.ndarray,
    detections: sv.Detections,
    threshold: float,
    class_names: dict,
) -> np.ndarray:
    filtered = filter_by_confidence(detections, threshold)
    labels = build_labels(filtered, class_names)
    box_annotator = sv.BoxAnnotator(thickness=3)
    label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.6)
    annotated = box_annotator.annotate(scene=image.copy(), detections=filtered)
    return label_annotator.annotate(
        scene=annotated, detections=filtered, labels=labels
    )


def image_to_bytes(img: Image.Image) -> bytes:
    import io

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


# ---------------------------------------------------------------------------
# ESTILOS
# ---------------------------------------------------------------------------

st.markdown(
    """
<style>
    .main { background-color: #1a1d23; }
    .block-container { max-width: 1400px !important; padding-top: 1.5rem !important; }

    .header {
        background: linear-gradient(135deg, #1a2332, #243447);
        padding: 2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        text-align: center;
        color: #e8edf5;
        box-shadow: 0 4px 20px rgba(0,0,0,0.3);
        border: 1px solid #2d3545;
    }
    .header h1 {
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.6rem;
        line-height: 1.35;
        letter-spacing: 0.5px;
    }
    .header .badge {
        display: inline-block;
        background: rgba(255,255,255,0.15);
        padding: 0.3rem 1.2rem;
        border-radius: 20px;
        font-size: 0.8rem;
        margin-bottom: 0.7rem;
        letter-spacing: 0.5px;
        border: 1px solid rgba(255,255,255,0.15);
        color: white;
        font-weight: 600;
    }
    .header .equipo {
        font-size: 1.5rem;
        font-weight: 400;
        margin-top: 1rem;
        border-top: 1px solid rgba(255,255,255,0.08);
        padding-top: 1rem;
        line-height: 2;
        text-align: left;
    }
    .header .equipo .linea {
        display: block;
        font-size: 1.55rem;
        font-weight: 500;
    }
    .header .equipo .cargo {
        font-weight: 700;
        color: #7ab7e0;
    }

    .card {
        background: #1a3050;
        padding: 1.5rem 2rem;
        border-radius: 14px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.25);
        margin-bottom: 1.2rem;
        border: 1px solid #2a4a6a;
    }
    .card-titulo {
        color: #b8d8f0;
        font-size: 1.1rem;
        font-weight: 600;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }

    .image-wrapper {
        background: #13233d;
        border: 1px solid #2a4a6a;
        border-radius: 12px;
        padding: 0.8rem;
        text-align: center;
    }
    .image-wrapper img {
        border-radius: 8px;
        width: 100%;
    }
    .image-label {
        font-size: 0.9rem;
        font-weight: 600;
        margin-bottom: 0.5rem;
        padding: 0.25rem 0.9rem;
        border-radius: 6px;
        display: inline-block;
    }
    .image-wrapper-original {
        background: #13233d;
        border: 1px solid #2a4a6a;
    }
    .image-wrapper-result {
        background: #1a2818;
        border: 1px solid #3a4a2a;
    }
    .image-label-original {
        background: #1a3050;
        color: #8dc4f0;
    }
    .image-label-result {
        background: #2a3a1a;
        color: #d4c84b;
    }

    .footer {
        text-align: center;
        color: #6a8aaa;
        font-size: 0.85rem;
        margin-top: 2rem;
        padding-top: 1rem;
        border-top: 1px solid #2a4a6a;
    }

    header[data-testid="stHeader"] { display: none !important; }
    .appview-container .main .block-container { padding-top: 1rem !important; }
    .stAlert { border-radius: 8px; }
    div[data-testid="stFileUploader"] { margin-bottom: 0; }

    p, ol, ul, dl { color: #d1d5db !important; }
    label, .stFileUploader { color: #d1d5db !important; }
    div[data-testid="stFileUploader"] section {
        padding: 2rem !important;
        border: 2px dashed #2a4a6a !important;
        border-radius: 14px !important;
        background: #0f1e33 !important;
    }
    div[data-testid="stFileUploader"] section:hover {
        border-color: #6da5d1 !important;
        background: #142a45 !important;
    }
    div[data-testid="stFileUploader"] section > div {
        color: #b8d8f0 !important;
        font-size: 1rem !important;
    }
    div[data-testid="column"] > div:has(button) { gap: 1rem; }
    .stButton button {
        background: #1a3050 !important;
        color: white !important;
        border: 1px solid #2a4a6a !important;
        padding: 0.5rem 1rem !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        transition: all 0.2s !important;
    }
    .stButton button:hover {
        background: #2a4a6a !important;
        border-color: #6da5d1 !important;
    }
    .stDownloadButton button {
        background: #c0392b !important;
        color: white !important;
        border: none !important;
        padding: 0.6rem 2.5rem !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.2s !important;
    }
    .stDownloadButton button:hover {
        background: #e74c3c !important;
        box-shadow: 0 2px 16px rgba(192,57,43,0.5) !important;
        transform: translateY(-1px) !important;
    }

    @media (max-width: 768px) {
        .header { padding: 1rem 1.2rem; }
        .header h1 { font-size: 1.2rem !important; }
        .header .badge { font-size: 0.7rem; padding: 0.2rem 0.8rem; color: white !important; background: rgba(255,255,255,0.15) !important; }
        .header .equipo { font-size: 0.8rem !important; }
        .header .equipo .linea { font-size: 0.9rem !important; }
        .card { padding: 0.8rem 1rem !important; }
        .card-titulo { font-size: 0.9rem !important; }
        .block-container { max-width: 100% !important; padding: 0.5rem 0.8rem !important; }
        .image-wrapper { padding: 0.4rem !important; }
        .image-label { font-size: 0.75rem !important; padding: 0.15rem 0.6rem !important; }
        div[data-testid="stFileUploader"] section { padding: 1rem !important; }
        .footer { font-size: 0.65rem !important; }
    }

    .modal-overlay {
        position: fixed;
        top: 0; left: 0; width: 100%; height: 100%;
        background: rgba(0,0,0,0.8);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 9999;
        padding: 1rem;
    }
    .modal-content {
        position: relative;
        max-width: 90%;
        max-height: 90%;
    }
    .modal-content img {
        max-width: 100%;
        max-height: 85vh;
        border-radius: 12px;
        box-shadow: 0 8px 40px rgba(0,0,0,0.5);
    }
    .modal-close {
        position: absolute;
        top: -40px;
        right: 0;
        background: rgba(255,255,255,0.15);
        color: white;
        border: none;
        font-size: 1.5rem;
        width: 36px;
        height: 36px;
        border-radius: 50%;
        cursor: pointer;
        display: flex;
        align-items: center;
        justify-content: center;
        transition: background 0.2s;
    }
    .modal-close:hover {
        background: rgba(255,255,255,0.3);
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# ENCABEZADO
# ---------------------------------------------------------------------------

st.markdown(
    """
<div class="header">
    <div class="badge">TESIS DE GRADO</div>
    <h1>MODELO DE DETECCIÓN Y SEGMENTACIÓN
    AUTOMÁTICA DE CARIES DENTAL CON
    RADIOGRAFÍAS PANORÁMICAS USANDO
    DEEP LEARNING</h1>
    <div class="equipo">
        <span class="linea"><span class="cargo">Postulante:</span> Luis Ronaldo Mamani Mayta</span>
        <span class="linea"><span class="cargo">Tutor Especialista:</span> Lic. Silvana Mayta Carrizales</span>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

if not validate_api_key():
    st.stop()

# ---------------------------------------------------------------------------
# MODAL / BOTONES
# ---------------------------------------------------------------------------

if "modal" not in st.session_state:
    st.session_state.modal = None

imgs = {"prob": "assets/problematica.jpg", "sol": "assets/solucion.jpg"}

if st.session_state.modal in imgs:
    st.markdown(
        f"""
    <style>
        .main > div {{ display: none !important; }}
        .modal-overlay {{ display: flex !important; }}
    </style>
    <div class="modal-overlay">
        <div class="modal-content">
            <img src="https://raw.githubusercontent.com/LuisRonaldo117/tesiscaries/main/assets/{st.session_state.modal}.jpg" />
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )
    if st.button("Cerrar", key="close_modal"):
        st.session_state.modal = None
        st.rerun()
    st.stop()

col_btn1, col_btn2, _ = st.columns([1, 1, 3])
with col_btn1:
    if st.button("Problematica", use_container_width=True):
        st.session_state.modal = "prob"
with col_btn2:
    if st.button("Solucion", use_container_width=True):
        st.session_state.modal = "sol"

# ---------------------------------------------------------------------------
# CARGA DE IMAGEN
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="card">'
    '<div class="card-titulo">Subir radiografía</div>',
    unsafe_allow_html=True,
)

uploaded_file = st.file_uploader(
    "Selecciona una radiografía panorámica o imagen dental",
    type=["jpg", "jpeg", "png", "bmp", "tiff"],
    label_visibility="collapsed",
)

st.markdown("</div>", unsafe_allow_html=True)

if uploaded_file is None:
    st.info("Sube una imagen para comenzar el análisis.")
    st.stop()

image = load_image(uploaded_file)
if image is None:
    st.stop()

with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
    tmp_path = Path(tmp.name)
    image.save(tmp_path, "JPEG")

result = run_inference(str(tmp_path), "1")
tmp_path.unlink(missing_ok=True)

if result is None:
    st.stop()

detections = sv.Detections.from_inference(result)
class_names = get_class_names(result)
image_array = np.array(image)
annotated_frame = annotate_image(
    image_array, detections, CONFIDENCE_THRESHOLD, class_names
)

# ---------------------------------------------------------------------------
# RESULTADOS
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="card">'
    '<div class="card-titulo">Resultado del análisis</div>',
    unsafe_allow_html=True,
)

col_a, col_b = st.columns(2)

with col_a:
    st.markdown(
        '<div class="image-wrapper image-wrapper-original">'
        '<div class="image-label image-label-original">Radiografía original</div>',
        unsafe_allow_html=True,
    )
    st.image(image, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with col_b:
    st.markdown(
        '<div class="image-wrapper image-wrapper-result">'
        '<div class="image-label image-label-result">Detección de caries</div>',
        unsafe_allow_html=True,
    )
    st.image(annotated_frame, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

filtered = filter_by_confidence(detections, CONFIDENCE_THRESHOLD)
if len(filtered) > 0:
    st.warning(
        f"Se detectaron {len(filtered)} zona(s) con posible caries dental."
    )
else:
    st.success("No se detectaron anomalías en la imagen analizada.")

st.download_button(
    label="Descargar imagen anotada",
    data=image_to_bytes(Image.fromarray(annotated_frame)),
    file_name="deteccion_caries.png",
    mime="image/png",
)

st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# PIE
# ---------------------------------------------------------------------------

st.markdown(
    '<div class="footer">'
    "Proyecto de Tesis &mdash; Carrera de Odontología &mdash; 2026"
    "</div>",
    unsafe_allow_html=True,
)
