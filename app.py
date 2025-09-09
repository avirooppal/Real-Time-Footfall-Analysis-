import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import tempfile
import os
import streamlit as st
from ultralytics import YOLO
import yt_dlp
import psutil
import subprocess

# Try GPU monitor
try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except:
    GPU_AVAILABLE = False

# Try lap vs lapx (for bytetrack)
try:
    import lap
except ImportError:
    import lapx as lap

# ----------------- Config -----------------
MODEL = "yolov8n.pt"  # lightweight model
HEATMAP_H, HEATMAP_W = 72, 128
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ----------------- Helper -----------------
def open_youtube_ffmpeg(url, width=640, height=360):
    """Open YouTube stream via yt-dlp + ffmpeg pipe."""
    ydl_opts = {"format": "best"}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        stream_url = info["url"]

    ffmpeg_cmd = [
        "ffmpeg",
        "-i", stream_url,
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-vf", f"scale={width}:{height}",
        "-"
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    return proc, width, height

def draw_overlay(frame, current_count, total_ids, fps):
    """Draw top overlay with stats."""
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 70), (0, 0, 0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
    cv2.putText(frame, f"Active: {current_count}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"Unique: {len(total_ids)}", (220, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"FPS: {fps:.1f}", (420, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return frame

# ----------------- Streamlit UI -----------------
st.set_page_config(layout="wide")
st.title("Footfall Analysis (YOLO)")

source_type = st.sidebar.radio("Select Source", ["YouTube Live", "Local File", "Webcam"])
if source_type == "YouTube Live":
    youtube_url = st.sidebar.text_input("YouTube URL",
                                        "https://www.youtube.com/watch?v=DjdUEyjx8GM")
elif source_type == "Local File":
    video_file = st.sidebar.file_uploader("Upload video", type=["mp4", "avi", "mov"])
else:
    webcam_index = st.sidebar.number_input("Webcam index", 0, 5, 0)

run_button = st.sidebar.button("Start Analysis")

# ----------------- Main Loop -----------------
if run_button:
    stframe = st.empty()
    heatmap_placeholder = st.empty()

    # Load YOLO model
    model = YOLO(MODEL)

    # Pick source
    proc = None
    cap = None
    if source_type == "YouTube Live":
        proc, w, h = open_youtube_ffmpeg(youtube_url)
    elif source_type == "Local File" and video_file:
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(video_file.read())
        cap = cv2.VideoCapture(tfile.name)
    else:
        cap = cv2.VideoCapture(int(webcam_index))

    heatmap = np.zeros((HEATMAP_H, HEATMAP_W), dtype=np.float32)
    total_ids = set()
    current_count = 0
    fps, last_time = 0, time.time()
    frame_idx = 0

    while True:
        # --- Read frame ---
        if proc:
            raw_frame = proc.stdout.read(w * h * 3)
            if not raw_frame:
                st.warning("⚠️ Stream ended or unavailable, reconnecting...")
                break
            frame = np.frombuffer(raw_frame, np.uint8).reshape((h, w, 3))
        else:
            ret, frame = cap.read()
            if not ret:
                st.warning("⚠️ Stream ended or unavailable, reconnecting...")
                break

        # --- YOLO tracking ---
        results = model.track(frame, tracker="bytetrack.yaml", persist=True, conf=0.35, imgsz=640)
        res = results[0]
        boxes = res.boxes
        ids_this_frame = set()

        for box in boxes:
            if int(box.cls[0]) != 0:  # only person class
                continue
            if box.id is None:
                continue
            track_id = int(box.id[0])
            ids_this_frame.add(track_id)
            total_ids.add(track_id)

            # centroid for heatmap
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            gx = int((cx / frame.shape[1]) * (HEATMAP_W - 1))
            gy = int((cy / frame.shape[0]) * (HEATMAP_H - 1))
            if frame_idx % 3 == 0:   # update less often
                heatmap[gy, gx] += 1

            # draw detection
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            cv2.putText(frame, f"ID:{track_id}", (int(x1), int(y1)-6),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        current_count = len(ids_this_frame)

        # --- FPS ---
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / (now - last_time))
        last_time = now
        frame_idx += 1

        # --- Overlay ---
        frame = draw_overlay(frame, current_count, total_ids, fps)

        # --- Display in Streamlit ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        stframe.image(frame_rgb, channels="RGB", use_container_width=True)

        # --- Resource usage ---
        cpu_percent = psutil.cpu_percent(interval=None)
        mem = psutil.virtual_memory()
        mem_percent = mem.percent

        col1, col2, col3 = st.columns(3)
        col1.metric("CPU Usage", f"{cpu_percent}%")
        col2.metric("RAM Usage", f"{mem_percent}%")
        if GPU_AVAILABLE:
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            gpu_percent = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
            col3.metric("GPU Usage", f"{gpu_percent}%")

        # --- Update heatmap every N frames ---
        if frame_idx % 30 == 0:
            heatmap_norm = heatmap / (heatmap.max() + 1e-6)
            heatmap_blur = cv2.GaussianBlur(heatmap_norm, (0, 0), sigmaX=2, sigmaY=2)
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.imshow(heatmap_blur, origin='lower', cmap='hot', interpolation='bilinear')
            ax.set_title("Footfall Heatmap")
            heatmap_placeholder.pyplot(fig)
            plt.close(fig)

    if cap:
        cap.release()
    if proc:
        proc.kill()
