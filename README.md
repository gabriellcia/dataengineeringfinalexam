# End-to-End Edge AI Pipeline: Urban Mobility & Emission Correlation

**Capstone Project — Data Engineering, CSIE, Tamkang University**

---

## 📌 Project Overview

This project implements an industrial-grade **Edge AI Gateway** that correlates real-time traffic density with air quality data. The system is designed to run on resource-constrained hardware (2-Core CPU), prioritizing architectural stability, multi-threaded efficiency, and fault-tolerant data engineering.

The pipeline ingests video frames and PM2.5 sensor readings simultaneously, performs edge inference, fuses the two streams via a temporal join, and aggregates the results into 30-second summaries before publishing to the cloud or caching locally on network failure.

---

## 🛠 Key Features

| Feature | Description |
|---|---|
| **Multi-threaded Architecture** | Separate Producer, Inference, and Consumer threads communicate via `queue.Queue(maxsize=1)` to prevent memory overflow (OOM) on constrained hardware. |
| **Heterogeneous Data Fusion** | Nearest-neighbor Temporal Join syncs PM2.5 sensor data (10 Hz) with YOLO vehicle detection (10 Hz) within a strict 100 ms tolerance window. |
| **Quantized Edge Inference** | Designed for TFLite INT8 models (simulated in the current build) to minimize CPU load during inference. |
| **Fault Tolerance & Resilience** | If MQTT transmission fails (simulated 20% failure rate), payloads are automatically written to a local JSONL fallback file (`local_storage.jsonl`) for deferred delivery. |
| **Sliding Window Aggregation** | Every 30 seconds, per-frame records are aggregated into a single summary payload (avg PM2.5, avg vehicle count, sample count, alert status), drastically reducing cloud bandwidth usage. |
| **CSV Performance Logging** | Every aggregation window is appended to `performance_metrics.csv`, enabling offline analysis in Excel or any BI tool. |

---

## 🏗 System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        MAIN PROCESS                             │
│                                                                 │
│  ┌──────────────┐    video_queue     ┌──────────────────────┐  │
│  │   PRODUCER   │ ──────(maxsize=1)─▶│    INFERENCE ENGINE  │  │
│  │  (Thread 1)  │                    │      (Thread 2)       │  │
│  │              │                    │                       │  │
│  │ • PM2.5 sim  │                    │ • Frame resize 224²   │  │
│  │   @ 10 Hz    │   sensor_buffer    │ • Vehicle detection   │  │
│  │ • MP4 frames │ ──(deque, 200)──┐  │   (INT8 simulation)   │  │
│  └──────────────┘                 │  └──────────┬────────────┘  │
│                                   │             │ fused_data_queue
│                                   │             ▼               │
│                                   │  ┌──────────────────────┐  │
│                                   └─▶│    DATA MANAGER      │  │
│                                      │     (Thread 3)        │  │
│                                      │                       │  │
│                                      │ • Temporal Join       │  │
│                                      │ • 30s aggregation     │  │
│                                      │ • CSV logging         │  │
│                                      │ • MQTT / JSONL        │  │
│                                      └──────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### Thread Responsibilities

**1. Producer (`producer_worker`)**
- Generates PM2.5 readings using a Gaussian distribution (μ=25.0, σ=5.0) at 10 Hz and pushes them into `sensor_buffer` (a `deque` with a rolling capacity of 200 readings).
- Reads frames from `sample_video.mp4` using OpenCV, looping back to frame 0 on video end.
- Applies a basic brightness quality check (`frame.mean() > 30`) before queuing a frame.

**2. Inference (`inference_worker`)**
- Pulls the latest frame from `video_queue`.
- Resizes it to 224×224 (standard model input size) using bilinear interpolation.
- Simulates INT8 model inference, producing a random vehicle count (2–12) for each frame.
- Forwards the timestamped result to `fused_data_queue`.

**3. Data Manager (`data_manager_worker`)**
- For each inference result, performs a **nearest-neighbor temporal join** against `sensor_buffer` to find the PM2.5 reading closest in time (within 100 ms).
- Accumulates fused records in `agg_buffer`.
- Every 30 seconds, computes window averages, triggers a `HIGH_POLLUTION` alert if `avg_pm25 > 35`, and calls `transmit_and_log`.

---

## 📊 Data Outputs

### `performance_metrics.csv`
Appended every 30-second aggregation window. Suitable for direct import into Excel or any analytics tool.

```
timestamp,avg_pm25,avg_vehicles,samples,status
06:05:13,25.06,7.2,282,NORMAL
06:05:43,24.94,7.1,283,NORMAL
...
```

| Column | Description |
|---|---|
| `timestamp` | Window start time (`HH:MM:SS`) |
| `avg_pm25` | Mean PM2.5 (μg/m³) over the window |
| `avg_vehicles` | Mean vehicle count per frame |
| `samples` | Number of fused records in the window |
| `status` | `NORMAL` or `HIGH_POLLUTION` (threshold: PM2.5 > 35) |

### `local_storage.jsonl`
Written only when MQTT transmission fails. Each line is a complete JSON summary payload, allowing deferred batch upload when connectivity is restored.

```json
{"summary_start": "01:54:58", "avg_pm25": 24.97, "avg_vehicles": 7.2, "samples": 283, "alert": "NORMAL"}
```

---

## 🚀 Setup & Installation

### Prerequisites
- Python 3.8+
- A video file named `sample_video.mp4` in the project root (any MP4 works; the pipeline loops it automatically)
- Optional: a running MQTT broker at `localhost:1883` (the pipeline simulates MQTT and falls back gracefully without one)

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/username/capstone-edge-ai.git
cd capstone-edge-ai

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the pipeline
python main_pipeline.py
```

The terminal will show rolling output from all three threads:

```
=== Edge AI Gateway Started (Logging to CSV) ===

[Producer]     Thread started using source: sample_video.mp4
[Inference]    Thread started (Simulation Mode)
[Data Manager] Thread started (Aggregation & CSV Logging)
[CLOUD]        MQTT Success: {'summary_start': '...', 'avg_pm25': 25.06, ...}
[LOCAL]        Network Down! Summary cached to local_storage.jsonl
```

Press `Ctrl+C` to shut down gracefully. All threads will finish their current work before exiting.

---

## 📦 Dependencies (`requirements.txt`)

```
# Core Edge AI & Computer Vision
opencv-python-headless   # Headless OpenCV (no GUI; suitable for servers/edge devices)
numpy                    # Vectorized frame processing
tflite-runtime           # Lightweight inference runtime (use tensorflow for non-edge)
pillow                   # Image utility support

# Data Engineering & Transport
paho-mqtt                # MQTT client for cloud pub/sub
protobuf                 # Serialization (used by MQTT/TFLite ecosystem)
jsonlines                # JSONL read/write helpers

# Utilities
psutil                   # System resource monitoring
urllib3                  # HTTP transport layer
```

> **Note:** On non-edge environments (e.g., development laptops), replace `tflite-runtime` with `tensorflow`. The inference thread is simulated in the current build and does not require either package to run.

---

## ⚙️ Configuration Reference

All tunable parameters are defined at the top of `main_pipeline.py`:

| Variable | Default | Description |
|---|---|---|
| `VIDEO_SOURCE` | `sample_video.mp4` | Path to the input video file |
| `MQTT_BROKER` | `localhost` | MQTT broker hostname |
| `MQTT_TOPIC` | `city/edge/urban_mobility` | MQTT publish topic |
| `FALLBACK_FILE` | `local_storage.jsonl` | Local cache for failed MQTT transmissions |
| `PERF_LOG_CSV` | `performance_metrics.csv` | CSV output for performance logging |
| `SYNC_TOLERANCE` | `0.100` | Max allowed drift for temporal join (seconds) |
| `WINDOW_SIZE` | `30` | Sliding window duration (seconds) |

---

## 📈 Performance Characteristics

- **Sync Error:** Guaranteed < 100 ms via nearest-neighbor temporal join across a 200-sample rolling sensor buffer.
- **Memory Safety:** `video_queue(maxsize=1)` ensures only the most recent frame is held in memory at any time, preventing unbounded growth on slow inference.
- **Throughput:** At 10 Hz with a 30-second window, each summary aggregates approximately 280–285 fused records (matching observed output in `performance_metrics.csv`).
- **Network Resilience:** Simulated 20% packet loss is handled transparently via JSONL fallback. No data is dropped.

---

## 📝 Submission Details

| Item | Specification |
|---|---|
| **Pitch Duration** | 5–8 minutes |
| **Video Demo** | 1–2 minute terminal & dashboard recording |
| **Deadline** | Submit slides and GitHub link 24 hours before the scheduled presentation |

---

## 🔮 Future Work

- Replace simulated inference with a real TFLite INT8 YOLOv10-N model for live vehicle detection.
- Add exponential backoff retry logic to the MQTT transmission path for more robust fault tolerance.
- Implement a Dead Letter Queue (DLQ) mechanism to track and replay failed payloads.
- Build a live dashboard (e.g., Grafana or Streamlit) fed from the CSV or MQTT broker.
- Extend the alert system with configurable PM2.5 thresholds and multi-level severity levels.
