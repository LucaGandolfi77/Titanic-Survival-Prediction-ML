"""
streamlit_app.py – Streamlit demo for PCB Defect Detection.

Features:
  1. Upload image / capture from webcam
  2. Adjust detection parameters (confidence, IoU, model size)
  3. View detections with bounding boxes
  4. Download results (annotated image + JSON)
  5. Batch upload mode
  6. Training metrics dashboard

Run:  streamlit run src/ui/streamlit_app.py --server.port 8503
"""
from __future__ import annotations

import io
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

# Add project root to path
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.utils.config import CLASS_NAMES, project_root, class_colours

# ── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title="🔍 PCB Defect Detection",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🔍 PCB Defect Detection — Real-Time Demo")
st.markdown("Upload a PCB image to detect manufacturing defects using **YOLOv8**.")

# ── Sidebar ──────────────────────────────────────────────────
st.sidebar.header("⚙️ Detection Settings")
confidence = st.sidebar.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05)
iou_threshold = st.sidebar.slider("IoU Threshold (NMS)", 0.1, 0.9, 0.45, 0.05)
img_size = st.sidebar.selectbox("Image Size", [320, 416, 512, 640], index=3)

model_option = st.sidebar.selectbox("Model", ["yolov8n", "yolov8s", "yolov8m"], index=1)
device_option = st.sidebar.selectbox("Device", ["cpu", "mps", "cuda:0"], index=0)

st.sidebar.markdown("---")
st.sidebar.markdown("""
### Defect Classes
| ID | Name |
|:--:|:-----|
| 0 | 🟠 scratch |
| 1 | 🔵 dent |
| 2 | 🟢 discoloration |
| 3 | 🔴 crack |
| 4 | 🟣 missing_component |
""")


# ── Model loading ────────────────────────────────────────────
@st.cache_resource
def load_predictor(model_name: str, device: str, conf: float, iou: float, imgsz: int):
    """Load YOLOv8 predictor (cached across reruns)."""
    from src.inference.predictor import DefectPredictor

    # Try trained weights first, then pretrained
    trained = project_root() / "models" / "trained" / "best.pt"
    weights = trained if trained.exists() else f"{model_name}.pt"
    return DefectPredictor(weights=weights, device=device, conf=conf, iou=iou, imgsz=imgsz)


predictor = load_predictor(model_option, device_option, confidence, iou_threshold, img_size)

# ── Tabs ─────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["📸 Single Image", "📂 Batch Upload", "📊 Metrics Dashboard", "ℹ️ About"])

# ── Tab 1: Single Image ─────────────────────────────────────
with tab1:
    col_up, col_res = st.columns([1, 1])

    with col_up:
        uploaded = st.file_uploader("Upload PCB image", type=["jpg", "jpeg", "png", "bmp"])
        use_sample = st.checkbox("Use sample image (synthetic)")

        image = None
        if uploaded:
            file_bytes = np.frombuffer(uploaded.read(), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            st.image(image[:, :, ::-1], caption="Uploaded Image", use_container_width=True)
        elif use_sample:
            # Generate a quick synthetic sample
            from src.data.synthetic_generator import SyntheticDefectGenerator
            gen = SyntheticDefectGenerator(output_dir=Path("/tmp/sample"), seed=int(time.time()) % 1000)
            image = gen._procedural_pcb()
            img_copy = image.copy()
            img_copy, _ = gen._scratch(img_copy)
            img_copy, _ = gen._dent(img_copy)
            image = img_copy
            st.image(image[:, :, ::-1], caption="Synthetic Sample", use_container_width=True)

    with col_res:
        if image is not None:
            if st.button("🚀 Detect Defects", type="primary", use_container_width=True):
                with st.spinner("Running inference…"):
                    result = predictor.predict_image(image, conf=confidence, iou=iou_threshold)

                st.success(f"✅ {result.count} defect(s) detected in {result.inference_ms:.1f} ms")

                # Draw and show annotated image
                from src.utils.visualization import draw_detections
                if result.detections:
                    boxes = np.array([[d.x1, d.y1, d.x2, d.y2] for d in result.detections])
                    classes = np.array([d.class_id for d in result.detections])
                    confs = np.array([d.confidence for d in result.detections])
                    annotated = draw_detections(image, boxes, classes, confs)
                    st.image(annotated[:, :, ::-1], caption="Detections", use_container_width=True)
                else:
                    st.image(image[:, :, ::-1], caption="No defects found", use_container_width=True)

                # Details table
                if result.detections:
                    import pandas as pd
                    det_data = [d.to_dict() for d in result.detections]
                    df = pd.DataFrame(det_data)
                    st.dataframe(df, use_container_width=True)

                # Download JSON
                result_json = json.dumps(result.to_dict(), indent=2)
                st.download_button("📥 Download JSON", result_json, "detections.json", "application/json")
        else:
            st.info("👆 Upload an image or enable synthetic sample to start.")

# ── Tab 2: Batch Upload ─────────────────────────────────────
with tab2:
    st.subheader("📂 Batch Defect Detection")
    batch_files = st.file_uploader("Upload multiple images", type=["jpg", "jpeg", "png"],
                                    accept_multiple_files=True, key="batch")

    if batch_files and st.button("🚀 Process Batch", key="batch_btn"):
        import pandas as pd
        progress = st.progress(0)
        all_results = []

        for i, f in enumerate(batch_files):
            file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            result = predictor.predict_image(img, conf=confidence)
            all_results.append({
                "file": f.name,
                "detections": result.count,
                "inference_ms": round(result.inference_ms, 1),
                "classes": ", ".join(d.class_name for d in result.detections),
            })
            progress.progress((i + 1) / len(batch_files))

        df = pd.DataFrame(all_results)
        st.dataframe(df, use_container_width=True)
        st.metric("Total Defects", df["detections"].sum())
        st.metric("Avg Inference", f"{df['inference_ms'].mean():.1f} ms")

# ── Tab 3: Metrics ───────────────────────────────────────────
with tab3:
    st.subheader("📊 Training & Evaluation Metrics")

    # Look for training results
    runs_dir = project_root() / "experiments" / "runs" / "train"
    if runs_dir.exists():
        run_dirs = sorted([d for d in runs_dir.iterdir() if d.is_dir()], reverse=True)
        if run_dirs:
            selected_run = st.selectbox("Select training run", [d.name for d in run_dirs])
            run_path = runs_dir / selected_run

            # Show training curves if results.csv exists
            results_csv = run_path / "results.csv"
            if results_csv.exists():
                import pandas as pd
                results = pd.read_csv(results_csv)
                results.columns = results.columns.str.strip()

                col1, col2 = st.columns(2)
                with col1:
                    if "train/box_loss" in results.columns:
                        st.line_chart(results[["train/box_loss", "train/cls_loss"]].rename(
                            columns={"train/box_loss": "Box Loss", "train/cls_loss": "Class Loss"}
                        ))
                with col2:
                    if "metrics/mAP50(B)" in results.columns:
                        st.line_chart(results[["metrics/mAP50(B)", "metrics/mAP50-95(B)"]].rename(
                            columns={"metrics/mAP50(B)": "mAP@0.5", "metrics/mAP50-95(B)": "mAP@0.5:0.95"}
                        ))

            # Show confusion matrix image if exists
            cm_img = run_path / "confusion_matrix.png"
            if cm_img.exists():
                st.image(str(cm_img), caption="Confusion Matrix")
        else:
            st.info("No training runs found. Train a model first.")
    else:
        st.info("No experiments directory found.")

# ── Tab 4: About ─────────────────────────────────────────────
with tab4:
    st.subheader("ℹ️ About This Project")
    st.markdown("""
    ### Real-Time PCB Defect Detection System

    **Architecture:** YOLOv8 (Ultralytics) — Anchor-free detection head  
    **Classes:** 5 defect types (scratch, dent, discoloration, crack, missing component)  
    **Pipeline:** Synthetic data → Training → ONNX export → FastAPI + Streamlit

    #### Features
    - 🔍 Single image & batch detection
    - 📹 Real-time webcam inference
    - 🌐 REST API (FastAPI) + WebSocket streaming
    - 📊 mAP evaluation & confusion matrix
    - 🐳 Dockerized deployment

    #### Tech Stack
    `YOLOv8` · `PyTorch` · `FastAPI` · `Streamlit` · `OpenCV` · `ONNX Runtime`
    """)
