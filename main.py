import streamlit as st

st.set_page_config(
page_title="DeepFake Detection",
page_icon="🔍",
layout="wide"
)

st.title("🔍 DeepFake Detection System")
st.write("AI-powered Real and Fake Video Detection")

st.markdown("""

### Welcome!

Upload a video to check whether it is REAL or FAKE.

This application uses a CNN-based deep learning model
for deepfake detection.
""")

st.info("Open the WebApp to start video detection.")
