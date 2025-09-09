# Real-Time Footfall Analysis with YOLOv8 and Streamlit

This project provides a comprehensive solution for real-time people counting and density analysis using the YOLOv8 object detection model. The application, built with Streamlit, can process video feeds from multiple sources (YouTube live streams, local files, webcams) to track individuals, calculate footfall metrics, and generate dynamic heatmaps of crowd movement.

-----

## Features

  * **Multi-Source Video Input:** Process video from YouTube live streams, local video files (`.mp4`, `.avi`, `.mov`), or live webcam feeds.
  * **Real-Time Object Tracking:** Utilizes YOLOv8 and ByteTrack for high-performance detection and tracking of individuals in the video feed.
  * **Footfall Statistics Dashboard:** Displays critical metrics in real-time:
      * **Active Count:** Number of individuals currently visible in the frame.
      * **Unique Count:** Total number of unique individuals detected since the analysis started.
      * **Processing FPS:** Frames Per Second, indicating analysis performance.
  * **Dynamic Footfall Heatmap:** Generates and continuously updates a heatmap overlay to visualize high-traffic areas and common pathways.
  * **System Resource Monitoring:** Includes a live dashboard to monitor CPU, RAM, and (if available) GPU utilization during processing.

-----

## Demo Preview

<img width="993" height="643" alt="image" src="https://github.com/user-attachments/assets/428700f0-0357-4560-ab69-8515a3107102" />
<img width="1027" height="736" alt="image" src="https://github.com/user-attachments/assets/0b6310b9-34a7-4af9-87ce-bed4f5ea3f88" />



-----

## Technology Stack

  * **Core Logic:** Python
  * **Object Detection/Tracking:** Ultralytics YOLOv8
  * **Web Dashboard:** Streamlit
  * **Video Processing:** OpenCV
  * **YouTube Stream Handling:** yt-dlp
  * **Data Visualization:** Matplotlib
  * **System Monitoring:** psutil, pynvml (for NVIDIA GPUs)

-----

## Installation Guide

Follow these steps to set up and run the project locally.

### 1\. Prerequisites

  * Python 3.8 or newer
  * NVIDIA GPU with CUDA installed (recommended for significantly better performance)

### 2\. Setup Process

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/avirooppal/Real-Time-Footfall-Analysis-.git
    cd Real-Time-Footfall-Analysis
    ```

2.  **Create a Virtual Environment:**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install Required Libraries:**
    Create a `requirements.txt` file with the following content:

    ```txt
    streamlit
    opencv-python-headless
    ultralytics
    yt-dlp
    matplotlib
    psutil
    pynvml # Optional: for NVIDIA GPU monitoring
    ```

    Then, install using pip:

    ```bash
    pip install -r requirements.txt
    ```

-----

## How to Use

1.  **Run the Streamlit Application:**
    Execute the following command in your terminal from the project root directory:

    ```bash
    streamlit run app.py
    ```

    *(Assuming you saved the provided code as `app.py`)*

2.  **Configure Input Source:**

      * Open the web application in your browser (usually `http://localhost:8501`).
      * In the sidebar, select the desired source type: "YouTube Live", "Local File", or "Webcam".
      * **For YouTube:** Paste the URL of the live stream into the text input field.
      * **For Local File:** Use the file uploader to select a video from your computer.
      * **For Webcam:** Ensure your webcam is connected and select the correct device index.

3.  **Start Analysis:**

      * Click the "Start Analysis" button in the sidebar.
      * The main area will display the processed video feed with bounding boxes and live statistics.
      * The heatmap and resource usage metrics will update dynamically below the video feed.

-----

## Configuration

You can adjust the core parameters directly in the script for different performance and accuracy trade-offs:

  * **YOLO Model Selection:**
    The script defaults to `yolov8n.pt` (nano), which is fast but less accurate. For higher accuracy at the cost of performance, change the `MODEL` variable to other pre-trained models like `yolov8s.pt` (small) or `yolov8m.pt` (medium).

    ```python
    # Change model size for accuracy vs. speed trade-off
    MODEL = "yolov8s.pt"  # From "yolov8n.pt"
    ```

  * **Heatmap Resolution:**
    The resolution of the heatmap grid can be adjusted by changing `HEATMAP_H` and `HEATMAP_W`. Higher values create a more detailed heatmap but may slightly increase processing overhead.

    ```python
    HEATMAP_H, HEATMAP_W = 72, 128
    ```

  * **Detection Confidence:**
    Modify the `conf` parameter in the `model.track()` call to filter out less confident detections. Lowering it (e.g., `0.25`) will detect more potential objects, while raising it (e.g., `0.5`) will result in fewer, higher-certainty detections.

    ```python
    results = model.track(frame, tracker="bytetrack.yaml", persist=True, conf=0.35)
    ```

