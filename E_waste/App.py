import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from PIL import Image
from pathlib import Path
import plotly.express as px


# ---------------------------------------
# Path Handling (Always Works in Cloud)
# ---------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR.parent / "models"

MODEL_PATH = MODEL_DIR / "ewaste_classifier.keras"
CLASS_INDEX_PATH = MODEL_DIR / "class_indices.json"


# ---------------------------------------
# Load Model with Fallback
# ---------------------------------------
def load_ai_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found at {MODEL_PATH}")
        st.stop()
    
    try:
        return load_model(str(MODEL_PATH), compile=False)
    except Exception as e:
        st.error("❌ Error loading model")
        st.error(str(e))
        st.stop()


model = load_ai_model()

# Load class labels
with open(CLASS_INDEX_PATH) as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

# Scrap prices dataset
scrap_prices = {
    "Battery": 300,
    "Keyboard": 150,
    "Microwave": 2500,
    "Mobile": 800,
    "Mouse": 80,
    "PCB": 200,
    "Player": 300,
    "Printer": 500,
    "Television": 1500
}


# ---------------------------------------
# Prediction Function
# ---------------------------------------
def predict_image(img: Image.Image):
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)[0]
    index = np.argmax(preds)

    label = idx_to_class[index]
    confidence = float(preds[index])
    price = scrap_prices[label]

    # Top 3 classes
    top3_idx = np.argsort(preds)[::-1][:3]
    top3 = {idx_to_class[i]: float(preds[i]) for i in top3_idx}

    return label, confidence, price, top3


# ---------------------------------------
# Streamlit Page Config
# ---------------------------------------
st.set_page_config(
    page_title="E-Waste Scrap Price Predictor",
    page_icon="♻️",
    layout="wide"
)

# Custom CSS for premium styling
st.markdown("""
<style>
body {
    background-color: #F4F6F9;
}
.header-title {
    font-size: 50px;
    text-align: center;
    font-weight: 900;
    padding-top: 10px;
    color: #2D3436;
}
.subtitle {
    text-align: center;
    font-size: 20px;
    margin-top: -10px;
    opacity: 0.8;
    margin-bottom: 40px;
}
.card-box {
    background: white;
    padding: 25px;
    border-radius: 14px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.07);
    margin: 10px 0px;
}
.footer {
    margin-top: 30px;
    text-align: center;
    opacity: 0.6;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)


# -----------------------------
# Sidebar Panel
# -----------------------------
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/566/566985.png", width=80)
st.sidebar.title("⚙️ Application Panel")
st.sidebar.markdown("---")

st.sidebar.subheader("📦 Model Information")
st.sidebar.info(f"Trained Classes: **{len(idx_to_class)}**")

st.sidebar.subheader("👨‍💻 Developer")
st.sidebar.write("**Ashwin Karthik**")

st.sidebar.markdown("---")
st.sidebar.caption("AI-powered E-Waste Analysis Tool")


# -----------------------------
# Tabs Navigation
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🏠 Home", "🔍 Predict", "ℹ️ About"])



# -----------------------------
# HOME TAB
# -----------------------------
with tab1:

    st.markdown('<div class="header-title">E-Waste Scrap Price Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Classify e-waste items & instantly estimate scrap price using AI</div>', unsafe_allow_html=True)

    banner_path = BASE_DIR / "banner.jpg"
    if banner_path.exists():
        st.image(str(banner_path), use_column_width=True)
    else:
        st.warning("⚠️ Banner image missing (banner.jpg)")


# -----------------------------
# PREDICT TAB
# -----------------------------
with tab2:

    st.header("📤 Upload or Capture Image")

    col_upload, col_camera = st.columns(2)

    with col_upload:
        uploaded_file = st.file_uploader(
            "Upload an e-waste image",
            type=["jpg", "jpeg", "png"]
        )

    with col_camera:
        st.write("Or capture using webcam:")
        camera_file = st.camera_input("Take a photo")

    # Determine which image to use
    img_source = None
    if uploaded_file:
        img_source = Image.open(uploaded_file).convert("RGB")
    elif camera_file:
        img_source = Image.open(camera_file).convert("RGB")

    if img_source:
        st.subheader("📸 Selected Image")
        st.image(img_source, use_container_width=True)

        st.subheader("🔍 Prediction")
        with st.spinner("Analyzing image..."):

            label, confidence, price, top_preds = predict_image(img_source)

            st.markdown(f"""
            <div class="card-box">
                <b>Category:</b> {label}<br>
                <b>Confidence:</b> {confidence*100:.2f}%<br>
                <b>Estimated Scrap Price:</b> ₹{price}
            </div>
            """, unsafe_allow_html=True)

        # Top 3 chart
        st.subheader("📊 Confidence Chart")
        fig = px.bar(
            x=list(top_preds.keys()),
            y=[p * 100 for p in top_preds.values()],
            labels={"x": "Category", "y": "Confidence (%)"},
            color=list(top_preds.keys()),
            title="Top 3 Predictions"
        )
        st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# ABOUT TAB
# -----------------------------
with tab3:

    st.header("ℹ️ About This Project")
    st.write("""
    This AI-powered application identifies **e-waste categories** from images and predicts the **scrap value** instantly.

    ### 🔥 Features
    - Smart e-waste item classification
    - Accurate scrap price estimation
    - Confidence-level visualization
    - Professional, responsive UI
    - Mobile-friendly interface

    ### 🧠 Technologies Used
    - TensorFlow (MobileNetV2)
    - Streamlit
    - Python & Deep Learning
    - Plotly for charts

    **Developer:** Ashwin Karthik  
    """)
# -----------------------------


st.markdown("<div class='footer'>© 2025 Made with ❤️ by Ashwin using Streamlit & TensorFlow</div>", unsafe_allow_html=True)
