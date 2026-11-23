import os
import numpy as np
from PIL import Image
from transformers import pipeline

MODEL_NAMES = [
    "Organika/sdxl-detector",
    "Heem2/AI-vs-Real-Image-Detection",
    "haywoodsloan/ai-image-detector-deploy",
]
AI_LABELS = {"fake", "artificial", "ai", "ai image", "deepfake", "ai-generated", "generated"}
EXTS = (".jpg", ".jpeg", ".png", ".webp")

REAL_DIR = os.path.join("test_images", "real")
AI_DIR = os.path.join("test_images", "ai")

# Folders khud bana do agar nahi hain
os.makedirs(REAL_DIR, exist_ok=True)
os.makedirs(AI_DIR, exist_ok=True)


def list_images(folder):
    files = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(EXTS)
    ]
    if not files:
        raise SystemExit(
            f"\nNo images found in '{folder}'.\n"
            f"Add some .jpg/.png/.webp files there and run again."
        )
    return files


real_files = list_images(REAL_DIR)
ai_files = list_images(AI_DIR)
print(f"Real images: {len(real_files)} | AI images: {len(ai_files)}\n")

if min(len(real_files), len(ai_files)) < 10:
    print("NOTE: 10 se kam images hain, isliye accuracy numbers bharosemand nahi honge.\n")


def ai_score(det, path):
    """AI/fake probability (0-1) for one image."""
    img = Image.open(path).convert("RGB")
    res = det(img, top_k=None)
    return sum(r["score"] for r in res if r["label"].strip().lower() in AI_LABELS)


def best_threshold(real, ai):
    """Threshold with the best balanced accuracy."""
    best = (0.5, 0.0)
    for t in np.arange(0.05, 1.0, 0.05):
        bal = ((real < t).mean() + (ai >= t).mean()) / 2
        if bal > best[1]:
            best = (float(t), float(bal))
    return best


results = {}
for name in MODEL_NAMES:
    try:
        det = pipeline("image-classification", model=name)
    except Exception as e:
        print(f"[SKIP] {name}: {e}\n")
        continue

    labels = list(det.model.config.id2label.values())
    print(f"== {name}")
    print(f"   labels: {labels}")
    if not any(l.strip().lower() in AI_LABELS for l in labels):
        print("   WARNING: no label matches AI_LABELS, so scores will always be 0. "
              "Add the right label to AI_LABELS.")

    try:
        real = np.array([ai_score(det, p) for p in real_files])
        ai = np.array([ai_score(det, p) for p in ai_files])
    except Exception as e:
        print(f"   [SKIP] error while scoring images: {e}\n")
        continue

    results[name] = (real, ai)

    acc_real = (real < 0.5).mean()
    acc_ai = (ai >= 0.5).mean()
    t, bal = best_threshold(real, ai)
    print(f"   @0.5  -> real correct: {acc_real:.0%} | AI correct: {acc_ai:.0%}")
    print(f"   best threshold: {t:.2f} (balanced accuracy {bal:.0%})")
    print(f"   avg score on real: {real.mean():.2f} | on AI: {ai.mean():.2f}\n")

if results:
    real_ens = np.mean([v[0] for v in results.values()], axis=0)
    ai_ens = np.mean([v[1] for v in results.values()], axis=0)
    t, bal = best_threshold(real_ens, ai_ens)
    print(f"== ENSEMBLE (average of {len(results)} models): "
          f"best threshold {t:.2f}, balanced accuracy {bal:.0%}")
else:
    print("No model could be evaluated.")