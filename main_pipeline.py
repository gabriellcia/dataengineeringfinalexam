import time
import threading
import queue
import json
import random
import os
import cv2
import numpy as np
from collections import deque
from datetime import datetime

# --- Integration of Lab 5 & 14 (Connectivity) ---
try:
    import paho.mqtt.client as mqtt
    HAS_MQTT = True
except ImportError:
    HAS_MQTT = False

# =================================================================
# GLOBAL CONFIGURATION (W1-W14 Standards)
# =================================================================
VIDEO_SOURCE   = "sample_video.mp4"
MQTT_BROKER    = "localhost"
MQTT_TOPIC     = "city/edge/urban_mobility"
FALLBACK_FILE  = "local_storage.jsonl"
PERF_LOG_CSV   = "performance_metrics.csv"  # File untuk Excel (Lab 1)
SYNC_TOLERANCE = 0.100  # 100ms max drift (Lab 14)
WINDOW_SIZE    = 30     # 30-second sliding window aggregation (Topic III)

# --- Initialize CSV Header if not exists (Lab 1: Bulk Write Prep) ---
if not os.path.exists(PERF_LOG_CSV):
    with open(PERF_LOG_CSV, "w") as f:
        f.write("timestamp,avg_pm25,avg_vehicles,samples,status\n")

# Thread-safe communication channels (Lab 13)
video_queue = queue.Queue(maxsize=1)   # Keeps only the freshest frame
sensor_buffer = deque(maxlen=200)       # Rolling buffer for scalar data
fused_data_queue = queue.Queue()        # Intermediate queue for fused results
stop_event = threading.Event()

# =================================================================
# 1. PRODUCER THREAD: SENSORS & VIRTUAL CAMERA (Lab 2, 7, 8)
# =================================================================
def producer_worker(video_path):
    """
    Simulates high-frequency sensor acquisition and frame grabbing.
    """
    cap = cv2.VideoCapture(video_path)
    print(f"[Producer] Thread started using source: {video_path}")
    
    while not stop_event.is_set():
        # A. Simulate Air Quality Sensor (PM2.5) @ 10Hz (Lab 2/8)
        pm_reading = round(random.normalvariate(25.0, 5.0), 2)
        sensor_buffer.append({"ts": time.time(), "value": pm_reading})
        
        # B. Grab Video Frame
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0) 
            continue
            
        # C. Visual Quality Control (Lab 7)
        if frame.mean() > 30: 
            if not video_queue.full():
                video_queue.put({"ts": time.time(), "frame": frame})
        
        time.sleep(0.1) # Maintain 10Hz frequency

# =================================================================
# 2. INFERENCE THREAD: EDGE AI ENGINE (Lab 9, 11, 13)
# =================================================================
def inference_worker():
    """
    Handles preprocessing and AI detection (YOLOv10-N / INT8).
    """
    print("[Inference] Thread started (Simulation Mode)")
    
    while not stop_event.is_set():
        try:
            data = video_queue.get(timeout=1)
            ts, frame = data["ts"], data["frame"]
            
            # --- Pre-processing (Lab 9/10 Vectorization) ---
            resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            
            # --- AI Inference Simulation (Lab 11) ---
            vehicle_count = random.randint(2, 12) 
            
            fused_data_queue.put({"ts": ts, "vehicle_count": vehicle_count})
            
        except queue.Empty:
            continue

# =================================================================
# 3. CONSUMER THREAD: FUSION & AGGREGATION (Lab 1, 4, 14)
# =================================================================
def data_manager_worker():
    """
    Executes Temporal Join, CSV Logging, and Cloud Exfiltration.
    """
    print("[Data Manager] Thread started (Aggregation & CSV Logging)")
    agg_buffer = []
    last_agg_timestamp = time.time()

    while not stop_event.is_set():
        try:
            inf_res = fused_data_queue.get(timeout=1)
            inf_ts = inf_res["ts"]
            
            # --- PART A: Heterogeneous Fusion (Lab 14) ---
            best_match = None
            min_drift = float('inf')
            
            for s in list(sensor_buffer):
                drift = abs(inf_ts - s["ts"])
                if drift < min_drift and drift <= SYNC_TOLERANCE:
                    min_drift = drift
                    best_match = s
            
            if best_match:
                record = {
                    "ts": inf_ts,
                    "pm25": best_match["value"],
                    "vehicle_count": inf_res["vehicle_count"]
                }
                agg_buffer.append(record)

                # --- PART B: Sliding Window Aggregation (Topic III Requirement) ---
                if time.time() - last_agg_timestamp >= WINDOW_SIZE:
                    if agg_buffer:
                        avg_pm = sum(r["pm25"] for r in agg_buffer) / len(agg_buffer)
                        avg_vc = sum(r["vehicle_count"] for r in agg_buffer) / len(agg_buffer)
                        
                        summary_payload = {
                            "summary_start": datetime.fromtimestamp(last_agg_timestamp).strftime('%H:%M:%S'),
                            "avg_pm25": round(avg_pm, 2),
                            "avg_vehicles": round(avg_vc, 1),
                            "samples": len(agg_buffer),
                            "alert": "HIGH_POLLUTION" if avg_pm > 35 else "NORMAL"
                        }
                        
                        # --- PART C: Logging to CSV & Cloud (Lab 1 & 4) ---
                        transmit_and_log(summary_payload)
                        
                        agg_buffer = []
                        last_agg_timestamp = time.time()

        except queue.Empty:
            continue

def transmit_and_log(payload):
    """Logs to CSV for Excel and handles Cloud transmission with Fallback"""
    
    # 1. Log to CSV for Excel Analysis (Every window) [cite: 131]
    with open(PERF_LOG_CSV, "a") as f:
        line = f"{payload['summary_start']},{payload['avg_pm25']},{payload['avg_vehicles']},{payload['samples']},{payload['alert']}\n"
        f.write(line)

    # 2. Simulate MQTT with Local Fallback (Lab 4/14) [cite: 40, 61, 132]
    is_network_up = random.random() > 0.2
    
    if is_network_up:
        print(f"[CLOUD] MQTT Success: {payload}")
    else:
        # Save to JSONL Fallback if cloud is "down" [cite: 40]
        with open(FALLBACK_FILE, "a") as f:
            f.write(json.dumps(payload) + "\n")
        print(f"[LOCAL] Network Down! Summary cached to {FALLBACK_FILE}")

# =================================================================
# MAIN EXECUTION
# =================================================================
if __name__ == "__main__":
    if not os.path.exists(VIDEO_SOURCE):
        print(f"[Error] Missing source video: {VIDEO_SOURCE}")
    else:
        producer = threading.Thread(target=producer_worker, args=(VIDEO_SOURCE,), daemon=True)
        inference = threading.Thread(target=inference_worker, daemon=True)
        manager = threading.Thread(target=data_manager_worker, daemon=True)

        print("\n=== Edge AI Gateway Started (Logging to CSV) ===\n")
        producer.start()
        inference.start()
        manager.start()

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n[Main] Shutdown signal received.")
            stop_event.set()
            producer.join()
            inference.join()
            manager.join()