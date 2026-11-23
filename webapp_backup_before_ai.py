import io
import os
import tempfile

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

try:  # registers the custom layers of the `efficientnet` package, if it was used for training
    import efficientnet.tfkeras  # noqa: F401
except Exception:
    pass

from tensorflow.keras.models import load_model

st.set_page_config(page_title="DeepFake Detector", page_icon="🔍", layout="wide")

st.title("🔍 DeepFake Detector")
st.write("Upload images or videos (or use the webcam) to check whether they are REAL or FAKE.")

MODEL_PATH = "tmp_checkpoint/best_model.keras"
IMG_SIZE = 128
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype="float32")
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype="float32")
IMAGE_EXT = {"jpg", "jpeg", "png"}
VIDEO_EXT = {"mp4", "avi", "mov", "mkv", "webm"}
MODES = ["none (0-255 raw)", "divide by 255", "ImageNet mean/std"]
AGG_OPTIONS = ["Mean", "Median", "Most-suspicious 20% of frames"]
MAX_SAMPLES_SHOWN = 6
ICONS = {"REAL": "✅", "FAKE": "❌", "UNCERTAIN": "❓"}


# ---------- Model ----------
@st.cache_resource
def load_detection_model():
    return load_model(MODEL_PATH)


try:
    model = load_detection_model()
    st.success("✅ Model loaded successfully")
except Exception as e:
    st.error("❌ Model could not be loaded")
    st.exception(e)
    st.stop()


# ---------- Face detector (optional) ----------
try:
    from mtcnn import MTCNN

    @st.cache_resource
    def get_detector():
        return MTCNN()

    MTCNN_OK = True
except Exception:
    MTCNN_OK = False


def crop_faces(pil_img, margin=0.3, min_conf=0.95, all_faces=False):
    """Same crop as training (30% margin, 95% confidence). Largest face first."""
    arr = np.array(pil_img)
    faces = [f for f in get_detector().detect_faces(arr) if f["confidence"] >= min_conf]
    if not faces:
        return []
    faces.sort(key=lambda f: f["box"][2] * f["box"][3], reverse=True)
    if not all_faces:
        faces = faces[:1]
    out = []
    for f in faces:
        x, y, w, h = f["box"]
        mx, my = int(w * margin), int(h * margin)
        x1, y1 = max(0, x - mx), max(0, y - my)
        x2, y2 = min(arr.shape[1], x + w + mx), min(arr.shape[0], y + h + my)
        crop = arr[y1:y2, x1:x2]
        if crop.size:
            out.append(Image.fromarray(crop))
    return out


def resize_like_training(pil_img):
    """Same resize as step 0: <300px 2x, 300-1000 1x, 1000-1900 0.5x, >1900 0.33x."""
    w, h = pil_img.size
    if w < 300:
        s = 2.0
    elif w <= 1000:
        s = 1.0
    elif w <= 1900:
        s = 0.5
    else:
        s = 0.33
    if s == 1.0:
        return pil_img
    return pil_img.resize((max(1, int(w * s)), max(1, int(h * s))))


def get_faces(pil_img, use_crop, all_faces, fallback_full):
    """Returns ([(128x128 array, face PIL), ...], face_found)."""
    pil_img = resize_like_training(pil_img)
    if use_crop and MTCNN_OK:
        crops = crop_faces(pil_img, all_faces=all_faces)
        if crops:
            return [(np.array(c.resize((IMG_SIZE, IMG_SIZE))), c) for c in crops], True
        if not fallback_full:
            return [], False
    return [(np.array(pil_img.resize((IMG_SIZE, IMG_SIZE))), pil_img)], False


# ---------- Preprocessing / prediction ----------
def preprocess(arr, mode):
    arr = arr.astype("float32")
    if mode == "divide by 255":
        return arr / 255.0
    if mode == "ImageNet mean/std":
        return (arr / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
    return arr  # none (0-255 raw)


def predict_batch(arrs, mode):
    """arrs: list of (128,128,3) uint8 arrays -> numpy array of REAL probabilities."""
    x = np.stack([preprocess(a, mode) for a in arrs])
    return model.predict(x, verbose=0, batch_size=32).reshape(-1)


def predict_chunked(X, mode, bar=None, chunk=64):
    out = []
    for i in range(0, len(X), chunk):
        out.append(predict_batch(list(X[i:i + chunk]), mode))
        if bar is not None:
            bar.progress(min(1.0, (i + chunk) / len(X)))
    return np.concatenate(out)


def extract_frames(path, n_frames):
    """Extract n_frames evenly spaced frames. Returns ([(frame_no, RGB array)], total_frames)."""
    cap = cv2.VideoCapture(path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    if total > 0:
        idxs = np.linspace(0, total - 1, min(n_frames, total)).astype(int)
        for i in idxs:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, frame = cap.read()
            if ok:
                frames.append((int(i), cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    return frames, total


def aggregate(p, method):
    p = np.asarray(p, dtype="float64")
    if method == "Median":
        return float(np.median(p))
    if method.startswith("Most"):
        k = max(1, int(np.ceil(0.2 * len(p))))
        return float(np.mean(np.sort(p)[:k]))
    return float(np.mean(p))


# ---------- Grad-CAM ----------
def _last_4d_layer(m):
    for layer in reversed(m.layers):
        try:
            if len(layer.output.shape) == 4:
                return layer
        except Exception:
            continue
    return None


@st.cache_resource
def build_cam_fn():
    """Returns f(x) -> (last conv feature map, model output), or None if not possible."""
    try:
        base = next((l for l in model.layers if isinstance(l, tf.keras.Model)), None)
        if base is not None:  # backbone is a nested model
            conv = _last_4d_layer(base)
            if conv is None:
                return None
            feat = tf.keras.Model(base.inputs, [conv.output, base.output])
            idx = model.layers.index(base)
            pre = [l for l in model.layers[:idx] if not isinstance(l, tf.keras.layers.InputLayer)]
            post = list(model.layers[idx + 1:])

            def fn(x):
                h = x
                for l in pre:
                    h = l(h)
                fmap, out = feat(h)
                h = out
                for l in post:
                    h = l(h)
                return fmap, h

            return fn
        conv = _last_4d_layer(model)
        if conv is None:
            return None
        gm = tf.keras.Model(model.inputs, [conv.output, model.output])
        return lambda x: gm(x)
    except Exception:
        return None


def gradcam_overlay(arr_uint8, mode):
    """Heatmap of the regions that push the model toward FAKE. Returns PIL image or None."""
    try:
        fn = build_cam_fn()
        if fn is None:
            return None
        x = tf.convert_to_tensor(np.expand_dims(preprocess(arr_uint8, mode), 0))
        with tf.GradientTape() as tape:
            fmap, pred = fn(x)
            target = 1.0 - pred[:, 0]  # FAKE score
        grads = tape.gradient(target, fmap)
        if grads is None:
            return None
        weights = tf.reduce_mean(grads, axis=(1, 2))
        cam = tf.reduce_sum(fmap * weights[:, None, None, :], axis=-1)[0]
        cam = tf.nn.relu(cam).numpy()
        cam = cam / cam.max() if cam.max() > 1e-8 else np.zeros_like(cam)
        cam = cv2.resize(cam.astype("float32"), (IMG_SIZE, IMG_SIZE))
        heat = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
        overlay = (0.55 * arr_uint8 + 0.45 * heat).astype(np.uint8)
        return Image.fromarray(overlay).resize((256, 256))
    except Exception:
        return None


# ---------- Robustness perturbations (applied on RGB uint8 arrays) ----------
def jpeg_rt(arr, q):
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    ok, enc = cv2.imencode(".jpg", bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(q)])
    if not ok:
        return arr
    return cv2.cvtColor(cv2.imdecode(enc, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)


def blur_fn(arr, k):
    return cv2.GaussianBlur(arr, (k, k), 0)


def noise_fn(arr, sigma):
    rng = np.random.default_rng(0)
    out = arr.astype("float32") + rng.normal(0, sigma, arr.shape)
    return np.clip(out, 0, 255).astype("uint8")


def rescale_fn(arr, f):
    h, w = arr.shape[:2]
    small = cv2.resize(arr, (max(8, int(w * f)), max(8, int(h * f))), interpolation=cv2.INTER_AREA)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)


PERTURBATIONS = {
    "Original": lambda a: a,
    "JPEG q=90": lambda a: jpeg_rt(a, 90),
    "JPEG q=70": lambda a: jpeg_rt(a, 70),
    "JPEG q=50": lambda a: jpeg_rt(a, 50),
    "JPEG q=30": lambda a: jpeg_rt(a, 30),
    "JPEG q=10": lambda a: jpeg_rt(a, 10),
    "Blur k=5": lambda a: blur_fn(a, 5),
    "Blur k=9": lambda a: blur_fn(a, 9),
    "Noise sigma=10": lambda a: noise_fn(a, 10),
    "Noise sigma=25": lambda a: noise_fn(a, 25),
    "Downscale 0.5x": lambda a: rescale_fn(a, 0.5),
    "Downscale 0.25x": lambda a: rescale_fn(a, 0.25),
    "WhatsApp-like (0.5x + JPEG 40)": lambda a: jpeg_rt(rescale_fn(a, 0.5), 40),
}


# ---------- Metrics (numpy only, REAL = positive class) ----------
def roc_auc(y, s):
    n_pos = int(np.sum(y == 1))
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    r = pd.Series(s).rank().values
    return float((r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def roc_curve_points(y, s):
    order = np.argsort(-s)
    ys = y[order]
    tps, fps = np.cumsum(ys), np.cumsum(1 - ys)
    tpr = np.r_[0.0, tps / max(tps[-1], 1)]
    fpr = np.r_[0.0, fps / max(fps[-1], 1)]
    return fpr, tpr, s[order]


def best_threshold(y, s):
    """Youden's J: threshold that maximises TPR - FPR."""
    fpr, tpr, s_sorted = roc_curve_points(y, s)
    j = (tpr - fpr)[1:]
    return float(s_sorted[int(np.argmax(j))])


def metrics_at(y, s, t):
    pred = (s >= t).astype(int)
    tp = int(np.sum((pred == 1) & (y == 1)))
    tn = int(np.sum((pred == 0) & (y == 0)))
    fp = int(np.sum((pred == 1) & (y == 0)))
    fn = int(np.sum((pred == 0) & (y == 1)))
    acc = (tp + tn) / max(len(y), 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-9)
    spec = tn / max(tn + fp, 1)
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, spec=spec, tp=tp, tn=tn, fp=fp, fn=fn)


def find_class_dir(root, name):
    for d in os.listdir(root):
        if d.lower() == name and os.path.isdir(os.path.join(root, d)):
            return os.path.join(root, d)
    return None


def list_images(folder, limit):
    files = [
        os.path.join(folder, f)
        for f in sorted(os.listdir(folder))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    if len(files) > limit:
        idx = np.linspace(0, len(files) - 1, limit).astype(int)
        files = [files[i] for i in idx]
    return files


def load_test_set(folder, per_class):
    """Returns (X, y, n_real, n_fake, error). y: real=1, fake=0."""
    if not os.path.isdir(folder):
        return None, None, 0, 0, f"Folder not found: {folder}"
    real_dir, fake_dir = find_class_dir(folder, "real"), find_class_dir(folder, "fake")
    if not real_dir or not fake_dir:
        return None, None, 0, 0, "The folder must contain subfolders named `real` and `fake`."
    files_r, files_f = list_images(real_dir, per_class), list_images(fake_dir, per_class)
    if not files_r or not files_f:
        return None, None, 0, 0, "No images (.jpg/.jpeg/.png) found in one of the folders."
    all_files = files_r + files_f
    y = np.r_[np.ones(len(files_r)), np.zeros(len(files_f))].astype(int)
    bar = st.progress(0.0, text="Loading images...")
    X = np.zeros((len(all_files), IMG_SIZE, IMG_SIZE, 3), dtype="uint8")
    for i, fp in enumerate(all_files):
        X[i] = np.array(Image.open(fp).convert("RGB").resize((IMG_SIZE, IMG_SIZE)))
        if i % 25 == 0:
            bar.progress(i / len(all_files), text=f"Loading images... {i}/{len(all_files)}")
    bar.empty()
    return X, y, len(files_r), len(files_f), None


# ---------- Callbacks ----------
def set_mode(m):
    st.session_state["mode"] = m


def set_threshold(t):
    st.session_state["threshold"] = float(min(0.95, max(0.05, round(t, 2))))


# ---------- Sidebar ----------
st.sidebar.header("Settings")
mode = st.sidebar.selectbox(
    "Preprocessing (must match training)",
    MODES,
    index=1,  # "divide by 255": training used rescale=1.0/255.0
    key="mode",
    help="Must match training (rescale=1/255 in 03-train_cnn.py). "
    "A wrong mode makes the model give the same answer every time.",
)
thr = st.sidebar.slider(
    "REAL cutoff",
    0.05,
    0.95,
    0.50,
    0.01,
    key="threshold",
    help="REAL probability >= cutoff -> REAL, otherwise FAKE. Raise it to catch more fakes (more false alarms).",
)
band = st.sidebar.slider(
    "Uncertain band (±)",
    0.0,
    0.30,
    0.10,
    0.01,
    help="If the REAL score is within this distance of the cutoff, the app says UNCERTAIN instead of guessing. Set 0 to turn off.",
)
use_crop = st.sidebar.checkbox("Face crop (MTCNN)", value=MTCNN_OK, disabled=not MTCNN_OK)
all_faces = st.sidebar.checkbox("Analyze all faces (not just the largest)", value=False, disabled=not MTCNN_OK)
n_frames = st.sidebar.slider("Video: number of frames to check", 5, 60, 20, step=5)
agg_method = st.sidebar.selectbox(
    "Video: how to combine frames",
    AGG_OPTIONS,
    help="Many deepfake videos are fake in only some frames. 'Most-suspicious 20%' stops the average from hiding them.",
)
min_face_ratio = st.sidebar.slider(
    "Video: min. share of frames with a face", 0.0, 1.0, 0.5, 0.05,
    help="If fewer frames than this contain a face, the app says CANNOT JUDGE instead of guessing.",
)
debug = st.sidebar.checkbox("Debug: show output for every mode", value=False)
if not MTCNN_OK:
    st.sidebar.warning("mtcnn is not installed: `pip install mtcnn`")

settings_sig = (mode, use_crop, all_faces, n_frames, debug)


def verdict_of(score):
    if abs(float(score) - thr) < band:
        return "UNCERTAIN"
    return "REAL" if score >= thr else "FAKE"


# ---------- Analysis ----------
def analyze_image(name, data):
    image = ImageOps.exif_transpose(Image.open(io.BytesIO(data))).convert("RGB")
    samples, found = get_faces(image, use_crop, all_faces, fallback_full=True)
    arrs = [a for a, _ in samples]
    faces = [f for _, f in samples]
    probs = predict_batch(arrs, mode)
    shown = list(range(min(len(arrs), MAX_SAMPLES_SHOWN)))
    res = {
        "name": name,
        "kind": "image",
        "probs": probs,
        "faces": faces,
        "frame_nos": [None] * len(faces),
        "sample_idx": shown,
        "heat": [gradcam_overlay(arrs[i], mode) for i in shown],
        "found": found,
        "skipped": 0,
        "total": None,
    }
    if debug:
        res["debug"] = {m: float(np.mean(predict_batch(arrs, m))) for m in MODES}
    return res


def analyze_video(name, data, ext):
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + ext) as tmp:
            tmp.write(data)
            tmp_path = tmp.name
        frames, total = extract_frames(tmp_path, n_frames)
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

    if not frames:
        return {"name": name, "kind": "video", "error": "Could not read frames from the video. Try another format (mp4)."}

    arrs, faces, frame_nos, skipped = [], [], [], 0
    bar = st.progress(0.0, text=f"{name}: detecting faces...")
    for k, (fno, rgb) in enumerate(frames):
        samples, found = get_faces(Image.fromarray(rgb), use_crop, all_faces, fallback_full=not (use_crop and MTCNN_OK))
        if use_crop and MTCNN_OK and not found:
            skipped += 1  # frames without a face are not counted
        else:
            for a, f in samples:
                arrs.append(a)
                faces.append(f)
                frame_nos.append(fno)
        bar.progress((k + 1) / len(frames), text=f"{name}: frame {k + 1}/{len(frames)}")
    bar.empty()

    if not arrs:
        return {"name": name, "kind": "video", "error": "No clear face found in any frame. Try a video with a visible face."}

    probs = predict_batch(arrs, mode)
    shown = np.linspace(0, len(arrs) - 1, min(MAX_SAMPLES_SHOWN, len(arrs))).astype(int).tolist()
    res = {
        "name": name,
        "kind": "video",
        "probs": probs,
        "faces": faces,
        "frame_nos": frame_nos,
        "sample_idx": shown,
        "heat": [gradcam_overlay(arrs[i], mode) for i in shown],
        "found": True,
        "skipped": skipped,
        "total": total,
        "face_ratio": (len(frames) - skipped) / len(frames),
    }
    if debug:
        res["debug"] = {m: float(np.mean(predict_batch(arrs, m))) for m in MODES}
    return res


def score_of(res):
    if res["kind"] == "image":
        return float(np.min(res["probs"]))  # image is FAKE if any face is FAKE
    return aggregate(res["probs"], agg_method)


# ---------- Result display ----------
def show_result(res, expanded):
    if "error" in res:
        with st.expander(f"⚠️ {res['name']}", expanded=True):
            st.warning(res["error"])
        return

    if res["kind"] == "video" and res.get("face_ratio", 1.0) < min_face_ratio:
        with st.expander(f"❓ {res['name']}: CANNOT JUDGE", expanded=True):
            st.warning(
                f"A face was found in only {res['face_ratio'] * 100:.0f}% of the sampled frames. "
                "This model only judges real human faces (face-swap deepfakes), so it cannot give "
                "a reliable answer for this video. It may be fully AI-generated or contain no real faces."
            )
        return

    score = score_of(res)
    v = verdict_of(score)
    icon = ICONS[v]
    probs = res["probs"]

    with st.expander(f"{icon} {res['name']}: {v} (REAL score {score * 100:.1f}%)", expanded=expanded):
        msg = f"{icon} {v}  |  REAL score {score * 100:.1f}%  |  FAKE score {(1 - score) * 100:.1f}%"
        {"REAL": st.success, "FAKE": st.error, "UNCERTAIN": st.warning}[v](msg)
        if v == "UNCERTAIN":
            st.caption(
                f"The score is within ±{band:.2f} of the cutoff ({thr:.2f}), so the app does not guess. "
                "Try a clearer, front-facing image, or more frames for a video."
            )

        if res["kind"] == "image":
            if use_crop and MTCNN_OK and not res["found"]:
                st.warning("No face detected, using the full image (result is less reliable).")
            if len(probs) > 1:
                st.caption(f"{len(probs)} faces found. The image is judged by its most suspicious face.")
        else:
            verdicts = [verdict_of(p) for p in probs]
            n_fake, n_unc = verdicts.count("FAKE"), verdicts.count("UNCERTAIN")
            st.write(
                f"**{n_fake} / {len(probs)}** samples FAKE ({n_fake / len(probs) * 100:.0f}%), "
                f"{n_unc} uncertain. {res['skipped']} frame(s) skipped (no face found). "
                f"Total video frames: {res['total']}. Combine method: {agg_method}."
            )
            st.subheader("Frame-by-frame REAL probability")
            st.line_chart(pd.DataFrame({"REAL probability": probs, "Cutoff": thr}))
            st.caption("Near 1 = REAL, near 0 = FAKE. A line that swings a lot means the model is uncertain.")

        st.subheader("Faces and Grad-CAM heatmaps")
        idxs = res["sample_idx"]
        cols = st.columns(len(idxs))
        for col, i in zip(cols, idxs):
            label = verdict_of(probs[i])
            who = f"Face {i + 1}" if res["kind"] == "image" else f"Frame {res['frame_nos'][i]}"
            col.image(res["faces"][i], caption=f"{who}: {probs[i] * 100:.0f}% real ({label})", use_container_width=True)
        cols = st.columns(len(idxs))
        for col, h in zip(cols, res["heat"]):
            if h is not None:
                col.image(h, caption="Heatmap", use_container_width=True)
            else:
                col.caption("Heatmap unavailable")
        st.caption(
            "Red/yellow areas pushed the model toward FAKE. If the heat sits on the background or "
            "hair edges instead of the face, the model may be using shortcuts."
        )

        if "debug" in res:
            st.subheader("Debug: all preprocessing modes (average)")
            st.table({"Mode": list(res["debug"]), "Avg REAL prob (1=real)": [round(x, 4) for x in res["debug"].values()]})


def build_reports(results):
    summary, detail = [], []
    for res in results:
        if "error" in res:
            summary.append({"File": res["name"], "Type": res["kind"], "Samples": 0, "REAL score %": None, "Verdict": "ERROR: " + res["error"]})
            continue
        if res["kind"] == "video" and res.get("face_ratio", 1.0) < min_face_ratio:
            summary.append({"File": res["name"], "Type": "video", "Samples": 0, "REAL score %": None, "Verdict": "CANNOT JUDGE (no face)"})
            continue
        s = score_of(res)
        summary.append(
            {
                "File": res["name"],
                "Type": res["kind"],
                "Samples": len(res["probs"]),
                "REAL score %": round(s * 100, 2),
                "Verdict": verdict_of(s),
            }
        )
        for i, p in enumerate(res["probs"]):
            detail.append(
                {
                    "File": res["name"],
                    "Type": res["kind"],
                    "Sample": i + 1,
                    "Frame": res["frame_nos"][i],
                    "REAL prob": round(float(p), 6),
                    "FAKE prob": round(float(1 - p), 6),
                    "Verdict": verdict_of(p),
                    "Cutoff": thr,
                    "Uncertain band": band,
                    "Preprocessing": res.get("mode", mode),
                }
            )
    return pd.DataFrame(summary), pd.DataFrame(detail)


# ---------- Tabs ----------
tab_detect, tab_eval, tab_robust, tab_ai = st.tabs(["🔍 Detect", "📊 Evaluation", "🧪 Robustness", "🎞️ AI-Video (CakeLens)"])

# ================= DETECT =================
with tab_detect:
    source = st.radio("Input", ["Upload files", "Webcam"], horizontal=True)
    items = []  # (name, bytes, kind, ext)

    if source == "Upload files":
        uploaded = st.file_uploader(
            "Upload images or videos (you can select several)",
            type=sorted(IMAGE_EXT | VIDEO_EXT),
            accept_multiple_files=True,
        )
        for f in uploaded or []:
            ext = f.name.rsplit(".", 1)[-1].lower()
            items.append((f.name, f.getvalue(), "image" if ext in IMAGE_EXT else "video", ext))
    else:
        cam = st.camera_input("Take a photo")
        if cam is not None:
            items.append(("webcam.png", cam.getvalue(), "image", "png"))

    files_sig = tuple((n, len(d)) for n, d, _, _ in items)

    if not items:
        st.info("Upload a file or take a photo to start.")
    else:
        if len(items) == 1:
            n, d, kind, _ = items[0]
            if kind == "image":
                st.image(ImageOps.exif_transpose(Image.open(io.BytesIO(d))), caption=n, width=360)
            else:
                st.video(d)
        else:
            st.caption(f"{len(items)} files selected.")

        if st.button(f"🔍 Analyze {len(items)} file(s)", type="primary"):
            results = []
            for n, d, kind, ext in items:
                with st.spinner(f"Analyzing {n}..."):
                    res = analyze_image(n, d) if kind == "image" else analyze_video(n, d, ext)
                    res["mode"] = mode
                    results.append(res)
            st.session_state["results"] = results
            st.session_state["files_sig"] = files_sig
            st.session_state["settings_sig"] = settings_sig

        results = st.session_state.get("results")
        if results and st.session_state.get("files_sig") == files_sig:
            if st.session_state.get("settings_sig") != settings_sig:
                st.warning("Mode / crop / frame settings changed since the last analysis. Click Analyze again to refresh. (Cutoff, uncertain band and combine method update live.)")

            st.divider()
            st.header("Results")
            summary_df, detail_df = build_reports(results)
            if len(results) > 1:
                st.dataframe(summary_df, use_container_width=True, hide_index=True)

            for res in results:
                show_result(res, expanded=(len(results) == 1))

            c1, c2 = st.columns(2)
            c1.download_button(
                "⬇️ Download summary CSV",
                summary_df.to_csv(index=False).encode("utf-8"),
                file_name="deepfake_summary.csv",
                mime="text/csv",
            )
            c2.download_button(
                "⬇️ Download frame-by-frame CSV",
                detail_df.to_csv(index=False).encode("utf-8"),
                file_name="deepfake_detail.csv",
                mime="text/csv",
            )

# ================= EVALUATION =================
with tab_eval:
    st.subheader("Model evaluation on a labeled test folder")
    st.write("Runs the model on real and fake images with known labels, and shows how accurate it really is.")
    folder = st.text_input("Test folder (must contain `real` and `fake` subfolders)", "split_dataset/test")
    per_class = st.slider("Max images per class", 50, 2000, 300, step=50)

    if st.button("▶️ Run evaluation"):
        X, y, n_r, n_f, err = load_test_set(folder, per_class)
        if err:
            st.error(err)
        else:
            probs = {}
            bar = st.progress(0.0, text="Predicting...")
            for m in MODES:
                bar.progress(0.0, text=f"Predicting: {m}")
                probs[m] = predict_chunked(X, m)
            bar.empty()
            st.session_state["eval"] = {"y": y, "probs": probs, "folder": folder, "n_real": n_r, "n_fake": n_f}

    ev = st.session_state.get("eval")
    if ev:
        y, all_probs = ev["y"], ev["probs"]
        st.success(f"Evaluated {ev['n_real']} real + {ev['n_fake']} fake images from `{ev['folder']}`")

        rows = []
        for m in MODES:
            s = all_probs[m]
            rows.append(
                {
                    "Preprocessing": m,
                    f"Accuracy @ {thr:.2f}": round(metrics_at(y, s, thr)["acc"], 4),
                    "ROC-AUC": round(roc_auc(y, s), 4),
                    "Avg REAL prob on real images": round(float(s[y == 1].mean()), 3),
                    "Avg REAL prob on fake images": round(float(s[y == 0].mean()), 3),
                }
            )
        table = pd.DataFrame(rows)
        st.subheader("Which preprocessing matches training?")
        st.dataframe(table, use_container_width=True, hide_index=True)
        st.caption("The correct mode has a high ROC-AUC, and a big gap between the last two columns (real images near 1, fake near 0).")

        aucs = [r["ROC-AUC"] for r in rows]
        best_mode = MODES[int(np.nanargmax(aucs))]
        best_auc = max(a for a in aucs if not np.isnan(a))
        if best_auc < 0.6:
            st.error("Even the best mode is close to random guessing (AUC < 0.6). The model may be undertrained, or the label mapping may differ.")
        st.button(f"✅ Use best mode: {best_mode}", on_click=set_mode, args=(best_mode,), disabled=(best_mode == mode))

        st.divider()
        st.subheader(f"Detailed metrics for the selected mode: {mode}")
        s = all_probs[mode]
        mt = metrics_at(y, s, thr)
        c = st.columns(5)
        c[0].metric("Accuracy", f"{mt['acc'] * 100:.1f}%")
        c[1].metric("Precision (REAL)", f"{mt['prec'] * 100:.1f}%")
        c[2].metric("Recall (REAL)", f"{mt['rec'] * 100:.1f}%")
        c[3].metric("Fake detection rate", f"{mt['spec'] * 100:.1f}%")
        c[4].metric("ROC-AUC", f"{roc_auc(y, s):.3f}")
        st.caption(f"F1 (REAL): {mt['f1']:.3f}  |  cutoff = {thr:.2f}  |  REAL is the positive class.")

        mask = np.abs(s - thr) >= band
        if band > 0 and mask.any() and not mask.all():
            sel = metrics_at(y[mask], s[mask], thr)
            st.info(
                f"With the uncertain band (±{band:.2f}), the model answers **{mask.mean() * 100:.0f}%** of images "
                f"and is **{sel['acc'] * 100:.1f}%** accurate on those answers "
                f"(vs {mt['acc'] * 100:.1f}% when forced to answer everything)."
            )

        left, right = st.columns(2)
        with left:
            st.write("**Confusion matrix**")
            cm = pd.DataFrame(
                [[mt["tp"], mt["fn"]], [mt["fp"], mt["tn"]]],
                index=["Actually REAL", "Actually FAKE"],
                columns=["Predicted REAL", "Predicted FAKE"],
            )
            st.table(cm)
        with right:
            st.write("**ROC curve**")
            fpr, tpr, _ = roc_curve_points(y, s)
            st.line_chart(pd.DataFrame({"False positive rate": fpr, "True positive rate": tpr}).set_index("False positive rate"))

        sug = best_threshold(y, s)
        st.write(f"Suggested cutoff for this mode (best balance of catching fakes and avoiding false alarms): **{sug:.2f}**")
        st.button(f"Apply cutoff {sug:.2f}", on_click=set_threshold, args=(sug,))

        st.caption(
            "If accuracy here is very high but the app does badly on your own images, the train/test split may "
            "contain frames from the same video in both sets (data leakage), or your images differ from the training data."
        )

# ================= ROBUSTNESS =================
with tab_robust:
    st.subheader("Robustness: how does the model behave on low-quality media?")
    st.write(
        "Videos shared on WhatsApp, Instagram or Telegram are compressed and resized. "
        "This tab applies such degradations and shows how the model's score and accuracy change."
    )
    st.caption(
        "Note: 'WhatsApp-like' is an approximation (downscale + JPEG), not the exact WhatsApp pipeline. "
        "Uses the preprocessing mode and cutoff selected in the sidebar."
    )
    sub1, sub2 = st.tabs(["Single image", "Test-folder benchmark"])

    # ----- single image -----
    with sub1:
        rob_file = st.file_uploader("Upload one image", type=sorted(IMAGE_EXT), key="rob_img")
        if rob_file is not None:
            rob_bytes = rob_file.getvalue()
            image = ImageOps.exif_transpose(Image.open(io.BytesIO(rob_bytes))).convert("RGB")
            samples, found = get_faces(image, use_crop, False, fallback_full=True)
            face_pil = samples[0][1].convert("RGB")
            base_arr = np.array(face_pil)
            st.image(face_pil, caption="Face used for the test", width=200)

            if st.button("🧪 Run stress test", key="rob_run1"):
                rows, thumbs = [], []
                for name, fn in PERTURBATIONS.items():
                    pert = fn(base_arr)
                    arr = np.array(Image.fromarray(pert).resize((IMG_SIZE, IMG_SIZE)))
                    p = float(predict_batch([arr], mode)[0])
                    rows.append((name, p))
                    thumbs.append((name, pert))
                st.session_state["rob1"] = {"sig": (rob_file.name, len(rob_bytes)), "rows": rows, "thumbs": thumbs}

            r1 = st.session_state.get("rob1")
            if r1 and r1["sig"] == (rob_file.name, len(rob_bytes)):
                orig_p = r1["rows"][0][1]
                orig_v = verdict_of(orig_p)
                table_rows = []
                for name, p in r1["rows"]:
                    table_rows.append(
                        {
                            "Perturbation": name,
                            "REAL prob %": round(p * 100, 1),
                            "Verdict": verdict_of(p),
                            "Change vs original (pp)": round((p - orig_p) * 100, 1),
                        }
                    )
                df1 = pd.DataFrame(table_rows)
                flips = int(sum(1 for r in table_rows[1:] if r["Verdict"] != orig_v))
                st.write(f"Original verdict: **{orig_v}**. The verdict changed in **{flips} of {len(table_rows) - 1}** perturbations.")
                st.dataframe(df1, use_container_width=True, hide_index=True)
                st.bar_chart(df1.set_index("Perturbation")["REAL prob %"])
                with st.expander("Show the perturbed images"):
                    cols = st.columns(4)
                    for k, (name, pert) in enumerate(r1["thumbs"]):
                        cols[k % 4].image(pert, caption=name, use_container_width=True)

    # ----- dataset benchmark -----
    with sub2:
        rob_folder = st.text_input("Test folder (needs `real` and `fake` subfolders)", "split_dataset/test", key="rob_folder")
        rob_per_class = st.slider("Max images per class", 50, 500, 100, step=50, key="rob_per_class")

        if st.button("▶️ Run robustness benchmark", key="rob_run2"):
            X, y, n_r, n_f, err = load_test_set(rob_folder, rob_per_class)
            if err:
                st.error(err)
            else:
                rows = []
                bar = st.progress(0.0, text="Running...")
                for k, (name, fn) in enumerate(PERTURBATIONS.items()):
                    bar.progress(k / len(PERTURBATIONS), text=f"Testing: {name}")
                    Xp = np.stack([fn(a) for a in X])
                    s = predict_chunked(Xp, mode)
                    mt = metrics_at(y, s, thr)
                    rows.append(
                        {
                            "Perturbation": name,
                            "Accuracy %": round(mt["acc"] * 100, 1),
                            "ROC-AUC": round(roc_auc(y, s), 3),
                            "Real recall %": round(mt["rec"] * 100, 1),
                            "Fake detection %": round(mt["spec"] * 100, 1),
                        }
                    )
                bar.empty()
                df = pd.DataFrame(rows)
                df["Accuracy drop (pp)"] = (df.loc[0, "Accuracy %"] - df["Accuracy %"]).round(1)
                st.session_state["rob2"] = {"df": df, "mode": mode, "thr": thr, "n_real": n_r, "n_fake": n_f}

        r2 = st.session_state.get("rob2")
        if r2:
            df = r2["df"]
            st.success(
                f"Benchmark on {r2['n_real']} real + {r2['n_fake']} fake images "
                f"(mode: {r2['mode']}, cutoff {r2['thr']:.2f})"
            )
            st.dataframe(df, use_container_width=True, hide_index=True)
            st.bar_chart(df.set_index("Perturbation")["Accuracy %"])
            worst = df.iloc[1:].sort_values("Accuracy %").iloc[0]
            st.write(
                f"Clean accuracy: **{df.loc[0, 'Accuracy %']}%**. Worst case: **{worst['Perturbation']}** "
                f"with **{worst['Accuracy %']}%** (drop of {worst['Accuracy drop (pp)']} points)."
            )
            st.download_button(
                "⬇️ Download robustness CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="robustness_results.csv",
                mime="text/csv",
            )

# ================= AI-GENERATED VIDEO (CakeLens v5) =================
# Separate pretrained model for fully AI-generated videos (Sora/Veo/Kling/...), not face-swaps.
# Install once:  pip install cakelens-v5 torch torchvision torchcodec huggingface-hub
AI_LABELS = [
    "AI_GEN", "ANIME_2D", "ANIME_3D", "VIDEO_GAME", "KLING", "HIGGSFIELD", "WAN", "MIDJOURNEY",
    "HAILUO", "RAY", "VEO", "RUNWAY", "SORA", "CHATGPT", "PIKA", "HUNYUAN", "VIDU",
]


@st.cache_resource(show_spinner="Loading CakeLens model (first time downloads ~3 GB)...")
def load_cakelens(device):
    from cakelens.detect import Detector
    from cakelens.model import Model

    m = Model()
    m.load_from_huggingface_hub()
    return Detector(model=m, batch_size=1, device=device)


with tab_ai:
    st.subheader("Is this video AI-generated? (CakeLens v5)")
    st.write(
        "Separate pretrained model for **fully AI-generated videos** (Sora, Veo, Kling, Runway, Pika, ...). "
        "It works on the whole frame, so it does not need a human face."
    )
    st.warning(
        "Limits: the authors report ~77% precision and ~74% recall (50% threshold) on their test set, and the model "
        "was trained on 2025 generators, so newer tools can fool it. Treat it as one clue, not proof."
    )
    ai_device = st.selectbox("Device", ["cpu", "cuda"], help="Use cuda only if you have an NVIDIA GPU with CUDA PyTorch.")
    ai_thr = st.slider("AI-generated cutoff", 0.05, 0.95, 0.50, 0.01)
    ai_file = st.file_uploader("Upload one video", type=sorted(VIDEO_EXT), key="ai_vid")

    if ai_file is not None:
        st.video(ai_file.getvalue())
        if st.button("🎞️ Check if AI-generated", type="primary", key="ai_run"):
            tmp_path = None
            try:
                import pathlib

                ext = ai_file.name.rsplit(".", 1)[-1].lower()
                with tempfile.NamedTemporaryFile(delete=False, suffix="." + ext) as tmp:
                    tmp.write(ai_file.getvalue())
                    tmp_path = tmp.name
                detector = load_cakelens(ai_device)
                with st.spinner("Analyzing video (can take a minute on CPU)..."):
                    verdict = detector.detect(pathlib.Path(tmp_path))
                probs = [float(p) for p in verdict.predictions]
                st.session_state["ai_res"] = {
                    "sig": (ai_file.name, len(ai_file.getvalue())),
                    "probs": probs,
                    "frames": getattr(verdict, "frame_count", None),
                }
            except ImportError:
                st.error("CakeLens is not installed. Run: pip install cakelens-v5 torch torchvision torchcodec huggingface-hub")
            except Exception as e:
                st.error("CakeLens failed to run.")
                st.exception(e)
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except OSError:
                        pass

        ar = st.session_state.get("ai_res")
        if ar and ar["sig"] == (ai_file.name, len(ai_file.getvalue())):
            p_ai = ar["probs"][0]  # label 0 = AI_GEN
            if p_ai >= ai_thr:
                st.error(f"❌ Likely AI-generated  |  AI_GEN score {p_ai * 100:.1f}%")
            else:
                st.success(f"✅ No strong sign of AI generation  |  AI_GEN score {p_ai * 100:.1f}%")
            if abs(p_ai - ai_thr) < 0.10:
                st.caption("The score is close to the cutoff, so treat this result as uncertain.")
            if ar["frames"] is not None:
                st.caption(f"Frames analyzed: {ar['frames']}")
            rows = [
                {"Label": AI_LABELS[i] if i < len(AI_LABELS) else f"Label {i}", "Probability %": round(p * 100, 2)}
                for i, p in enumerate(ar["probs"])
            ]
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
            st.caption("Other labels guess the style/generator and are less accurate than AI_GEN.")

st.caption(
    "These scores are only the model's output, not proof. The model works frame by frame "
    "(it does not look at motion/temporal cues) and can be wrong on new deepfake methods."
)