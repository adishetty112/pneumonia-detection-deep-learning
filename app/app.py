import streamlit as st
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from src.predict import predict
from src.gradcam import make_gradcam

st.set_page_config(page_title="Pneumonia Detection", page_icon="🫁")
st.title("🫁 Pneumonia Detection from Chest X-ray")
st.caption("EfficientNet-B0 transfer-learning demonstration")

uploaded = st.file_uploader("Upload a chest X-ray", type=["jpg","jpeg","png"])
if uploaded:
    os.makedirs("results", exist_ok=True)
    path = "results/uploaded_xray.png"
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    st.image(path, caption="Uploaded X-ray", use_container_width=True)

    if os.path.exists("models/pneumonia_efficientnet.pth"):
        label, confidence = predict(path)
        st.subheader(f"Prediction: {label}")
        st.metric("Model confidence", f"{confidence*100:.2f}%")
        try:
            make_gradcam(path, "results/gradcam.jpg")
            st.image("results/gradcam.jpg", caption="Grad-CAM visualization", use_container_width=True)
        except Exception as e:
            st.warning(f"Grad-CAM unavailable: {e}")
    else:
        st.error("Model not found. Run: python src/train.py")
