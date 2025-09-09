import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import os
from ultralytics import YOLO
import yt_dlp   # NEW: for YouTube live streams

# ---------- Config ----------
# VIDEO_SOURCE = "mall_entrance.mp4"   # local file or 0 for webcam
YOUTUBE_URL = "https://www.youtube.com/watch?v=DjdUEyjx8GM"

MODEL = "yolov8n.pt"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

HEATMAP_H, HEATMAP_W = 72, 128   # coarse heatmap grid

# ---------- Init ----------
model = YOLO(MODEL)

def get_youtube_stream(url):
    """Resolve YouTube URL into a direct stream link."""
    ydl_opts = {"format": "best"}
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        return info["url"]

# Pick source
# cap = cv2.VideoCapture(VIDEO_SOURCE)
stream_url = get_youtube_stream(YOUTUBE_URL)
cap = cv2.VideoCapture(stream_url)

# Prepare heatmap + counts
heatmap = np.zeros((HEATMAP_H, HEATMAP_W), dtype=np.float32)
total_ids = set()
current_count = 0

# FPS calc
last_time = time.time()
fps = 0

# Create resizable window
cv2.namedWindow("Mall Footfall Showcase", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Mall Footfall Showcase", 1280, 720)

print("Press 'q' to quit | 's' to save heatmap")

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        print("⚠️ Stream ended or dropped. Reconnecting...")
        cap.release()
        stream_url = get_youtube_stream(YOUTUBE_URL)
        cap = cv2.VideoCapture(stream_url)
        continue

    results = model.track(frame, tracker="bytetrack.yaml", persist=True, conf=0.35)
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

        # centroid
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        gx = int((cx / frame.shape[1]) * (HEATMAP_W - 1))
        gy = int((cy / frame.shape[0]) * (HEATMAP_H - 1))
        if frame_idx % 3 == 0:   # update heatmap less often (faster)
            heatmap[gy, gx] += 1

        # draw person box + ID
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"ID:{track_id}", (int(x1), int(y1)-6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    current_count = len(ids_this_frame)

    # --- FPS calc ---
    now = time.time()
    fps = 0.9*fps + 0.1*(1.0/(now - last_time))
    last_time = now
    frame_idx += 1

    # --- Overlay labels ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (0,0), (frame.shape[1], 70), (0,0,0), -1)
    frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

    label_text = [
        f"Current Active: {current_count}",
        f"Unique Visitors: {len(total_ids)}",
        f"FPS: {fps:.1f}",
        "Press 's' to save heatmap | 'q' to quit"
    ]
    for i, txt in enumerate(label_text):
        cv2.putText(frame, txt, (20, 30 + i*25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    cv2.imshow("Mall Footfall Showcase", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('s'):
        print("Saving heatmap...")
        heatmap_norm = heatmap / (heatmap.max() + 1e-6)
        heatmap_blur = cv2.GaussianBlur(heatmap_norm, (0,0), sigmaX=2, sigmaY=2)
        plt.figure(figsize=(8,6))
        plt.imshow(heatmap_blur, origin='lower', cmap='hot', interpolation='bilinear')
        plt.colorbar(label="Footfall density")
        plt.title("Mall Footfall Heatmap")
        plt.savefig(os.path.join(OUTPUT_DIR, "heatmap.png"), dpi=150)
        plt.close()
        print(f"Heatmap saved to {OUTPUT_DIR}/heatmap.png")

cap.release()
cv2.destroyAllWindows()
