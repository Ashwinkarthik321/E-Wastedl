import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import json
from PIL import Image
import plotly.express as px

# -----------------------------
# Load Model & Classes
# -----------------------------
MODEL_PATH = "models/ewaste_classifier.h5"
CLASS_INDEX_PATH = "models/class_indices.json"

model = load_model(MODEL_PATH)

with open(CLASS_INDEX_PATH) as f:
    class_indices = json.load(f)

idx_to_class = {v: k for k, v in class_indices.items()}

# Scrap prices
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


# -----------------------------
# Prediction Function
# -----------------------------
def predict_image(img):
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    pred = model.predict(arr)[0]  # probabilities
    index = np.argmax(pred)

    label = idx_to_class[index]
    confidence = float(pred[index])
    price = scrap_prices[label]

    # top 3 predictions
    sorted_idx = np.argsort(pred)[::-1][:3]
    top_preds = {idx_to_class[i]: float(pred[i]) for i in sorted_idx}

    return label, confidence, price, top_preds


# -----------------------------
# STYLING
# -----------------------------
st.set_page_config(page_title="E-Waste Scrap Price Predictor", page_icon="♻️", layout="wide")

st.markdown("""
    <style>
    .title {
        font-size: 45px;
        font-weight: 900;
        text-align: center;
        padding-top: 10px;
        color: #4a4a4a;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        margin-bottom: 50px;
        opacity: 0.8;
    }
    .card {
        background-color: #f7f9fb;
        padding: 20px;
        border-radius: 12px;
        border: 1px solid #e5e5e5;
        box-shadow: 0px 2px 10px rgba(0,0,0,0.05);
        margin-bottom: 15px;
    }
    .footer {
        font-size: 14px;
        text-align: center;
        margin-top: 40px;
        opacity: 0.7;
    }
    </style>
""", unsafe_allow_html=True)


# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.title("⚙️ Settings")

st.sidebar.write("**Model Info**")
st.sidebar.info(f"Classes: {len(class_indices)} total")

st.sidebar.write("**Developer**")
st.sidebar.markdown("🧑‍💻 **Ashwin Karthik**")

st.sidebar.write("**Scrap Price Source**")
st.sidebar.caption("Static dataset values (modifiable)")

st.sidebar.write("---")
st.sidebar.write("✨ *Professional UI Powered by Streamlit*")


# -----------------------------
# Tabs Navigation
# -----------------------------
tab1, tab2, tab3 = st.tabs(["🏠 Home", "🔍 Predict", "📄 About"])


# -----------------------------
# HOME TAB
# -----------------------------
with tab1:
    st.markdown('<div class="title">E-Waste Category & Scrap Price Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">AI-powered tool to classify e-waste & estimate scrap price instantly</div>', unsafe_allow_html=True)

    st.image("https://images.unsplash.com/photo-1581091012184-5c6a83090536", use_container_width=True, caption="Electronic Waste Recycling")


# -----------------------------
# PREDICT TAB
# -----------------------------
with tab2:

    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader("Upload an e-waste image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        img = Image.open(uploaded_file).convert("RGB")

        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("Uploaded Image")
            st.image(img, use_container_width=True)

        with col2:
            st.subheader("Prediction")
            with st.spinner("Analyzing image... 🔍"):

                label, confidence, price, top_preds = predict_image(img)

                st.markdown(f"""
                <div class="card">
                    <b>Category:</b> {label}<br>
                    <b>Confidence:</b> {confidence*100:.2f}%<br>
                    <b>Estimated Scrap Price:</b> ₹{price}
                </div>
                """, unsafe_allow_html=True)

                # Probability Chart
                st.subheader("Top Predictions")
                fig = px.bar(
                    x=list(top_preds.keys()),
                    y=[p * 100 for p in top_preds.values()],
                    labels={"x": "Category", "y": "Confidence (%)"},
                    color=list(top_preds.keys()),
                    title="Top 3 Prediction Probabilities"
                )
                st.plotly_chart(fig, use_container_width=True)


# -----------------------------
# ABOUT TAB
# -----------------------------
with tab3:
    st.header("📄 About This Project")
    st.write("""
    This AI-powered application classifies various e-waste products using deep learning 
    and estimates scrap price based on predefined pricing data.

    **Features:**
    - Image classification using MobileNetV2 (Transfer Learning)
    - Scrap price estimation
    - Confidence score with probability graph
    - Modern professional UI with tabs & charts

    **Technologies Used:**
    - Python, TensorFlow, Keras
    - Streamlit
    - Plotly (charts)
    - Deep Learning (CNNs)

    **Developer:** Ashwin Karthik
    """)

# Footer
st.markdown("<div class='footer'>© 2025 E-Waste Scrap Price Predictor • Built with ❤️ using Streamlit & TensorFlow</div>", unsafe_allow_html=True)
