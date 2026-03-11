from __future__ import annotations

import json
import time
import threading
from collections import deque
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

# =====================
# Page config
# =====================
st.set_page_config(page_title="Tool Classifier Realtime", page_icon="🛠️", layout="wide")

# Ưu tiên deploy_model (bỏ augmentation) để realtime mượt và chuẩn deploy
CANDIDATES = [
    "runs/tools_mnv2/deploy_model.keras",
    "runs/tools_mnv2/best_model.keras",
]
DEFAULT_MODEL_PATH = next((p for p in CANDIDATES if Path(p).exists()), CANDIDATES[0])
DEFAULT_LABELS_PATH = "runs/tools_mnv2/class_names.json"

RTC_CONFIG = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

PRED_LOCK = threading.Lock()


@st.cache_resource
def load_model_and_labels(model_path: str, labels_path: str):
    model_file = Path(model_path)
    labels_file = Path(labels_path)

    if not model_file.exists():
        raise FileNotFoundError(f"Không tìm thấy model: {model_file}")
    if not labels_file.exists():
        raise FileNotFoundError(f"Không tìm thấy labels: {labels_file}")

    model = tf.keras.models.load_model(str(model_file))
    class_names = json.loads(labels_file.read_text(encoding="utf-8"))

    img_size = model.input_shape[1] if model.input_shape and model.input_shape[1] else 160
    return model, class_names, int(img_size)


def preprocess_bgr_frame(frame_bgr: np.ndarray, img_size: int) -> np.ndarray:
    """
    - Center-crop về hình vuông (giảm méo)
    - Resize về img_size
    - Trả về (1, img_size, img_size, 3) float32 trong [0..255]
    """
    h, w = frame_bgr.shape[:2]
    side = min(h, w)
    y0 = (h - side) // 2
    x0 = (w - side) // 2
    crop = frame_bgr[y0:y0 + side, x0:x0 + side]

    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_AREA)

    x = rgb.astype(np.float32)
    x = np.expand_dims(x, axis=0)
    return x


def predict_one(model: tf.keras.Model, x: np.ndarray) -> np.ndarray:
    # model đã có preprocess Rescaling bên trong (deploy_model)
    with PRED_LOCK:
        probs = model(x, training=False).numpy()[0]
    return probs


def show_training_artifacts(run_dir: Path):
    st.subheader("📉 Training artifacts (loss/acc + confusion matrix + report)")

    loss_png = run_dir / "loss_plot.png"
    acc_png = run_dir / "acc_plot.png"
    cm_png = run_dir / "confusion_matrix.png"
    report_txt = run_dir / "classification_report.txt"
    metrics_json = run_dir / "test_metrics.json"

    cols = st.columns(2)
    with cols[0]:
        if loss_png.exists():
            st.image(str(loss_png), caption="Loss (train/val)", use_container_width=True)
        else:
            st.info("Chưa thấy loss_plot.png trong run_dir")

    with cols[1]:
        if acc_png.exists():
            st.image(str(acc_png), caption="Accuracy (train/val)", use_container_width=True)
        else:
            st.info("Chưa thấy acc_plot.png trong run_dir")

    if cm_png.exists():
        st.image(str(cm_png), caption="Confusion Matrix (test)", use_container_width=True)

    if metrics_json.exists():
        st.write("**Test metrics:**")
        st.json(json.loads(metrics_json.read_text(encoding="utf-8")))

    if report_txt.exists():
        st.write("**Classification report:**")
        st.code(report_txt.read_text(encoding="utf-8"))


def make_video_processor(model, class_names, img_size: int, infer_every_n: int, smooth_window: int, conf_thresh: float, mirror: bool):
    class ToolVideoProcessor(VideoProcessorBase):
        def __init__(self):
            self.model = model
            self.class_names = class_names
            self.img_size = img_size
            self.infer_every_n = max(1, int(infer_every_n))
            self.conf_thresh = float(conf_thresh)
            self.mirror = bool(mirror)

            self.frame_i = 0
            self.last_label = "..."
            self.last_conf = 0.0

            self.idx_hist = deque(maxlen=max(1, int(smooth_window)))

            self._fps = 0.0
            self._prev_t = time.time()

        def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
            img = frame.to_ndarray(format="bgr24")

            if self.mirror:
                img = cv2.flip(img, 1)

            # FPS estimate
            now = time.time()
            dt = now - self._prev_t
            self._prev_t = now
            if dt > 0:
                inst_fps = 1.0 / dt
                self._fps = inst_fps if self._fps == 0 else (0.9 * self._fps + 0.1 * inst_fps)

            self.frame_i += 1

            if (self.frame_i % self.infer_every_n) == 0:
                x = preprocess_bgr_frame(img, self.img_size)
                probs = predict_one(self.model, x)

                idx = int(np.argmax(probs))
                conf = float(probs[idx])

                # smoothing theo majority vote idx (giảm nhảy label)
                self.idx_hist.append(idx)
                idx_smooth = max(set(self.idx_hist), key=self.idx_hist.count)
                conf_smooth = float(probs[idx_smooth])

                if conf_smooth < self.conf_thresh:
                    self.last_label = "KHÔNG CHẮC"
                    self.last_conf = conf_smooth
                else:
                    self.last_label = str(self.class_names[idx_smooth])
                    self.last_conf = conf_smooth

            # Overlay
            text1 = f"{self.last_label}  ({self.last_conf*100:.1f}%)"
            text2 = f"FPS: {self._fps:.1f} | infer_every={self.infer_every_n}"

            cv2.putText(img, text1, (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(img, text2, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

            return av.VideoFrame.from_ndarray(img, format="bgr24")

    return ToolVideoProcessor


def predict_image_pil(model: tf.keras.Model, img: Image.Image, img_size: int) -> np.ndarray:
    img = img.convert("RGB").resize((img_size, img_size))
    x = np.array(img).astype(np.float32)
    x = np.expand_dims(x, 0)
    probs = predict_one(model, x)
    return probs


def main():
    st.title("🛠️ Tool Classifier — Realtime Webcam")
    st.write("Nhận diện: **Búa / Cờ lê / Kìm / Tua vít** (classification)")

    st.sidebar.header("Cấu hình")
    model_path = st.sidebar.text_input("MODEL_PATH", DEFAULT_MODEL_PATH)
    labels_path = st.sidebar.text_input("LABELS_PATH", DEFAULT_LABELS_PATH)

    infer_every_n = st.sidebar.slider("Infer mỗi N frame (tăng FPS)", 1, 10, 3)
    smooth_window = st.sidebar.slider("Smoothing window (vote)", 1, 10, 5)
    conf_thresh = st.sidebar.slider("Ngưỡng tin cậy (dưới ngưỡng => KHÔNG CHẮC)", 0.0, 1.0, 0.60, 0.01)
    mirror = st.sidebar.checkbox("Mirror webcam", value=True)

    if st.sidebar.button("Xoá cache model"):
        st.cache_resource.clear()
        st.rerun()

    try:
        model, class_names, img_size = load_model_and_labels(model_path, labels_path)
    except Exception as e:
        st.error(str(e))
        st.stop()

    st.sidebar.markdown("### Danh sách lớp")
    st.sidebar.write(class_names)
    st.sidebar.markdown(f"**IMG_SIZE:** `{img_size}`")

    tab1, tab2, tab3 = st.tabs(["📺 Realtime Webcam", "🖼️ Upload/Chụp ảnh", "📊 Train Metrics"])

    with tab1:
        st.subheader("📺 Nhận diện Realtime qua Webcam")
        st.caption("Hệ thống sẽ dự đoán trực tiếp trên khung hình và hiển thị nhãn cùng độ tin cậy theo thời gian thực.")

        VideoProcessor = make_video_processor(
            model=model,
            class_names=class_names,
            img_size=img_size,
            infer_every_n=infer_every_n,
            smooth_window=smooth_window,
            conf_thresh=conf_thresh,
            mirror=mirror,
        )

        webrtc_streamer(
            key="tool-realtime",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=RTC_CONFIG,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )

        st.info(
            "**Hướng dẫn vận hành thực nghiệm:**\n"
            "- Đặt dụng cụ ở vùng trung tâm camera, đảm bảo điều kiện ánh sáng tốt để đạt độ chính xác cao nhất.\n"
            "- Nếu nhãn bị nhảy (không ổn định): điều chỉnh thanh 'Smoothing window' hoặc tăng ngưỡng tin cậy ở cột bên trái.\n"
            "- Để tối ưu tốc độ xử lý (FPS): tăng thông số 'Infer mỗi N frame' (giảm tần suất dự đoán của model)."
        )

    with tab2:
        st.subheader("🖼️ Kiểm thử trên ảnh tĩnh")
        col1, col2 = st.columns([1, 1])

        with col1:
            option = st.radio("Nguồn dữ liệu đầu vào:", ("Sử dụng Webcam", "Tải tệp từ máy tính"), horizontal=True)
            uploaded = st.camera_input("Chụp ảnh kiểm thử") if option == "Sử dụng Webcam" else st.file_uploader(
                "Chọn tệp ảnh...", type=["jpg", "jpeg", "png", "webp"]
            )

        with col2:
            if uploaded is None:
                st.info("Vui lòng chọn hoặc chụp ảnh để hệ thống thực hiện dự đoán.")
            else:
                image = Image.open(uploaded)
                st.image(image, caption="Mẫu dữ liệu kiểm thử", use_container_width=True)

                probs = predict_image_pil(model, image, img_size)
                idx = int(np.argmax(probs))
                label = class_names[idx]
                conf = float(probs[idx])

                st.success(f"Kết quả phân loại: **{label}** — Độ tự tin: **{conf*100:.2f}%**")

                topk = np.argsort(probs)[::-1][:3]
                st.write("Top-3 xác suất cao nhất:")
                for k in topk:
                    st.write(f"- {class_names[int(k)]}: {float(probs[int(k)])*100:.2f}%")

                try:
                    import pandas as pd
                    df = pd.DataFrame({"class": class_names, "prob": probs}).set_index("class")
                    st.bar_chart(df)
                except Exception:
                    st.write("Thống kê xác suất phân lớp:")
                    for i, name in enumerate(class_names):
                        st.write(f"- {name}: {float(probs[i])*100:.2f}%")

    with tab3:
        run_dir = Path(model_path).resolve().parent
        st.subheader("📊 Phân tích kết quả huấn luyện")
        show_training_artifacts(run_dir)

    st.caption("Đồ án: Ứng dụng Transfer Learning với MobileNetV2 cho bài toán phân loại dụng cụ cơ khí.")


if __name__ == "__main__":
    main()