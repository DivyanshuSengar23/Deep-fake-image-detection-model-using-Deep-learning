import streamlit as st
import torch
from PIL import Image
from transformers import AutoImageProcessor, AutoModelForImageClassification

st.set_page_config(page_title="AI Image Detector", page_icon="🔍")
st.title("🔍 AI & Deepfake Detection System")
st.write("Upload an image to analyze whether it is REAL or AI-GENERATED.")

@st.cache_resource
def load_detector():
    name = "delpot/steganograph-ia-detector"
    processor = AutoImageProcessor.from_pretrained(name, use_fast=False)
    model = AutoModelForImageClassification.from_pretrained(name)
    model.eval()
    return processor, model

uploaded = st.file_uploader("Upload image", type=["jpg","jpeg","png","webp"])

if uploaded:
    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("Analyzing image..."):
        processor, model = load_detector()
        inputs = processor(images=image, return_tensors="pt")

        with torch.inference_mode():
            logits = model(**inputs).logits
            probs = torch.softmax(logits, dim=1)[0]

        labels = model.config.id2label
        ai_index = next(i for i, label in labels.items()
                        if label.lower() == "ai_generated")
        real_index = next(i for i, label in labels.items()
                          if label.lower() == "real")

        ai_score = probs[ai_index].item()
        real_score = probs[real_index].item()

    st.subheader("Detection Result")
    col1, col2 = st.columns(2)
    col1.metric("AI Generated", f"{ai_score*100:.2f}%")
    col2.metric("Real Photograph", f"{real_score*100:.2f}%")

    if ai_score >= 0.5:
        st.error("❌ FAKE / AI-GENERATED IMAGE")
    else:
        st.success("✅ LIKELY REAL PHOTOGRAPH")

    st.progress(ai_score)
    st.caption("Prediction is probabilistic, not guaranteed. Results depend on image quality and generator.")
