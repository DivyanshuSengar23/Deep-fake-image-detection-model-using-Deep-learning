import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
from huggingface_hub import hf_hub_download
import csv

REPO_ID = "nihal4/Deep_Fake_Hybrid_Model"
DATASET = Path("split_dataset")

print("Loading model...")
model_path = hf_hub_download(
    repo_id=REPO_ID,
    filename="deepfake_hybrid_final.onnx"
)
hf_hub_download(
    repo_id=REPO_ID,
    filename="deepfake_hybrid_final.onnx.data"
)

session = ort.InferenceSession(
    model_path,
    providers=["CPUExecutionProvider"]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

extensions = {".jpg", ".jpeg", ".png", ".webp"}
results = []

for path in DATASET.rglob("*"):
    if not path.is_file() or path.suffix.lower() not in extensions:
        continue

    actual = path.parent.name.lower()
    if actual not in ("fake", "real"):
        continue

    try:
        img = Image.open(path).convert("RGB").resize((260, 260))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - MEAN) / STD
        arr = arr.transpose(2, 0, 1)[None, ...].astype(np.float32)

        logits = session.run([output_name], {input_name: arr})[0]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        probs = exp / exp.sum(axis=1, keepdims=True)

        fake_score = float(probs[0][0] * 100)
        real_score = float(probs[0][1] * 100)
        prediction = "fake" if fake_score > real_score else "real"

        correct = prediction == actual

        results.append([
            str(path), actual, round(fake_score, 2),
            round(real_score, 2), prediction, correct
        ])

        print(
            f"{path.name}: Actual={actual.upper()}, "
            f"Predicted={prediction.upper()}, "
            f"FAKE={fake_score:.2f}%, REAL={real_score:.2f}%"
        )

    except Exception as e:
        print(f"Error: {path} -> {e}")

with open("batch_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow([
        "Image", "Actual", "Fake Score",
        "Real Score", "Prediction", "Correct"
    ])
    writer.writerows(results)

total = len(results)
correct = sum(row[5] for row in results)

print("\n========== FINAL REPORT ==========")
print("Total images:", total)
print("Correct predictions:", correct)
print("Incorrect predictions:", total - correct)

if total:
    print(f"Overall accuracy: {correct / total * 100:.2f}%")

for label in ("fake", "real"):
    subset = [r for r in results if r[1] == label]
    if subset:
        good = sum(r[5] for r in subset)
        print(f"{label.upper()}: {good}/{len(subset)} correct")

print("\nResults saved to batch_results.csv")
