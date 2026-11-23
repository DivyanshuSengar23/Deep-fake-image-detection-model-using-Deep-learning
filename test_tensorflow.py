import csv
from pathlib import Path
import numpy as np
import tensorflow as tf
from PIL import Image

model = tf.keras.models.load_model(
    "tmp_checkpoint/best_model.keras", compile=False
)

results = []

for label in ["fake", "real"]:
    folder = Path("prepared_dataset") / label

    for path in folder.glob("*"):
        if path.suffix.lower() not in [".png", ".jpg", ".jpeg"]:
            continue

        image = Image.open(path).convert("RGB").resize((128, 128))
        x = np.asarray(image, dtype=np.float32) / 255.0
        x = np.expand_dims(x, axis=0)

        real_score = float(model.predict(x, verbose=0)[0][0])
        prediction = "real" if real_score >= 0.5 else "fake"

        results.append({
            "Image": path.name,
            "Actual": label,
            "Real%": round(real_score * 100, 2),
            "Fake%": round((1 - real_score) * 100, 2),
            "Prediction": prediction,
            "Correct": prediction == label
        })

with open("tensorflow_results.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=results[0].keys())
    writer.writeheader()
    writer.writerows(results)

correct = sum(r["Correct"] for r in results)

print("\n===== TENSORFLOW FINAL REPORT =====")
print("Total:", len(results))
print("Correct:", correct)
print("Incorrect:", len(results) - correct)
print("Accuracy:", round(correct / len(results) * 100, 2), "%")

for label in ["fake", "real"]:
    subset = [r for r in results if r["Actual"] == label]
    hits = sum(r["Correct"] for r in subset)
    print(label.upper(), ":", hits, "/", len(subset), "correct")

print("CSV saved: tensorflow_results.csv")
