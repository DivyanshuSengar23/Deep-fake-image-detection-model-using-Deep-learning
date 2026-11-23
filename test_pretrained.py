import numpy as np
import onnxruntime as ort
from PIL import Image
from huggingface_hub import hf_hub_download

REPO_ID = "nihal4/Deep_Fake_Hybrid_Model"

print("Downloading model...")

model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="deepfake_hybrid_final.onnx"
)

hf_hub_download(
    repo_id=REPO_ID,
    filename="deepfake_hybrid_final.onnx.data"
)

print("Loading model...")

session = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def predict(path):
    image = Image.open(path).convert("RGB").resize((260, 260))
    arr = np.asarray(image, dtype=np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = arr.transpose(2, 0, 1)[None, ...].astype(np.float32)

    logits = session.run(
        [output_name],
        {input_name: arr}
    )[0]

    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)

    print("\n===== DETECTION RESULT =====")
    print(f"FAKE: {probs[0][0] * 100:.2f}%")
    print(f"REAL: {probs[0][1] * 100:.2f}%")
    print("Prediction:", "FAKE" if probs[0][0] > probs[0][1] else "REAL")

if __name__ == "__main__":
    path = input("Image path: ").strip().strip('"')
    predict(path)
