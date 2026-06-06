import time
import threading
import queue
import json
import random
import os
import math
import csv
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
VIDEO_SOURCE    = "sample_video.mp4"
MQTT_BROKER     = "localhost"
MQTT_TOPIC      = "city/edge/urban_mobility"
FALLBACK_FILE   = "local_storage.jsonl"
PERF_LOG_CSV    = "performance_metrics.csv"
LATENCY_LOG_CSV = "latency_profiling.csv"       # NEW: Jitter profiling (Hint #2)
SYNC_TOLERANCE  = 0.100   # 100ms max drift (Lab 14)
WINDOW_SIZE     = 30      # 30-second sliding window aggregation (Topic III)
RUN_DURATION    = 120     # Auto-stop after 120 seconds (2 menit untuk presentasi)
ANOMALY_INTERVAL = 45     # NEW: inject PM2.5 spike every 45s (Topic I hint)
POWER_SAVE_THRESHOLD = 3  # NEW: skip inference if vehicle delta < 3 (Topic II hint)

# --- Initialize CSV Headers if not exists ---
if not os.path.exists(PERF_LOG_CSV):
    with open(PERF_LOG_CSV, "w") as f:
        f.write("timestamp,avg_pm25,avg_vehicles,samples,status,anomaly_flag\n")

# NEW: Latency profiling log (Hint #2: Latency Jitter Profiling)
if not os.path.exists(LATENCY_LOG_CSV):
    with open(LATENCY_LOG_CSV, "w") as f:
        f.write("timestamp,sync_error_ms,vehicle_count,pm25,status\n")

# Thread-safe communication channels (Lab 13)
video_queue      = queue.Queue(maxsize=1)   # Keeps only the freshest frame
sensor_buffer    = deque(maxlen=200)         # Rolling buffer for scalar data
fused_data_queue = queue.Queue()             # Intermediate queue for fused results
stop_event       = threading.Event()

# NEW: Shared state for dashboard & power-saving
dashboard_state = {
    "frames_dropped": 0,
    "frames_processed": 0,
    "total_sync_errors": 0,
    "sync_error_samples": [],
    "last_summary": None,
    "power_save_skips": 0,
    "anomalies_injected": 0,
    "cloud_sent": 0,
    "fallback_cached": 0,
    "lock": threading.Lock()
}
state_lock = dashboard_state["lock"]

# =================================================================
# HELPER: Terminal Dashboard (NEW - untuk keperluan presentasi)
# =================================================================
def print_dashboard():
    """Prints a live summary panel to terminal every 10 seconds."""
    while not stop_event.is_set():
        time.sleep(10)
        with state_lock:
            s = dashboard_state
            avg_jitter = (
                sum(s["sync_error_samples"]) / len(s["sync_error_samples"]) * 1000
                if s["sync_error_samples"] else 0
            )
            last = s["last_summary"] or {}

        print("\n" + "="*60)
        print("  📊  EDGE AI GATEWAY — LIVE DASHBOARD")
        print("="*60)
        print(f"  Frames Processed : {s['frames_processed']}")
        print(f"  Frames Dropped   : {s['frames_dropped']}  (queue full → OOM prevention)")
        print(f"  Power-Save Skips : {s['power_save_skips']}  (inference suppressed)")
        print(f"  Avg Sync Jitter  : {avg_jitter:.2f} ms  (target: <100ms)")
        print(f"  Anomalies Inject : {s['anomalies_injected']}  (PM2.5 spike events)")
        print(f"  Cloud MQTT Sent  : {s['cloud_sent']}")
        print(f"  Local Fallback   : {s['fallback_cached']}")
        if last:
            print(f"\n  Last 30s Window:")
            print(f"    avg_pm25     = {last.get('avg_pm25', '-')}")
            print(f"    avg_vehicles = {last.get('avg_vehicles', '-')}")
            print(f"    samples      = {last.get('samples', '-')}")
            print(f"    alert        = {last.get('alert', '-')}")
        print("="*60 + "\n")

# =================================================================
# NEW: PM2.5 Sensor Replay from CSV (Hint #1 Method B)
# =================================================================
def load_pm25_dataset(filepath="pm25_dataset.csv"):
    """
    Loads a real-world PM2.5 CSV for playback.
    Falls back to mathematical simulation if file not found.
    Returns a list of float values.
    """
    if os.path.exists(filepath):
        values = []
        with open(filepath, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    values.append(float(row.get("pm25") or row.get("value") or 0))
                except ValueError:
                    continue
        if values:
            print(f"[Producer] Loaded {len(values)} PM2.5 records from CSV (Method B)")
            return values
    print("[Producer] No CSV dataset found — using mathematical simulation (Method A)")
    return None

# =================================================================
# 1. PRODUCER THREAD: SENSORS & VIRTUAL CAMERA (Lab 2, 7, 8)
# =================================================================
def producer_worker(video_path):
    """
    Simulates high-frequency sensor acquisition and frame grabbing.
    NEW: Supports CSV playback (Method B) + anomaly injection.
    """
    cap = cv2.VideoCapture(video_path)
    print(f"[Producer] Thread started using source: {video_path}")

    # NEW: Try loading real CSV dataset (Method B, Hint #1)
    pm25_dataset = load_pm25_dataset()
    csv_index = 0

    tick = 0  # frame counter for anomaly scheduling

    while not stop_event.is_set():
        tick += 1

        # A. Simulate / Replay PM2.5 Sensor @ 10Hz
        if pm25_dataset:
            # Method B: replay real CSV row-by-row
            pm_reading = round(pm25_dataset[csv_index % len(pm25_dataset)], 2)
            csv_index += 1
        else:
            # Method A: mathematical simulation (original code preserved)
            pm_reading = round(random.normalvariate(25.0, 5.0), 2)

        # NEW: Anomaly Injection — spike PM2.5 every ANOMALY_INTERVAL seconds
        # Simulates a real pollution event correlated with traffic (Topic I hint adapted)
        if tick % (ANOMALY_INTERVAL * 10) == 0:  # *10 because 10Hz
            pm_reading = round(random.uniform(55.0, 80.0), 2)  # HIGH_POLLUTION spike
            with state_lock:
                dashboard_state["anomalies_injected"] += 1
            print(f"[Producer] ⚠️  ANOMALY INJECTED — PM2.5 spike: {pm_reading} µg/m³")

        sensor_buffer.append({"ts": time.time(), "value": pm_reading})

        # B. Grab Video Frame
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        # C. Visual Quality Control (Lab 7) — original preserved
        if frame.mean() > 30:
            if not video_queue.full():
                video_queue.put({"ts": time.time(), "frame": frame})
            else:
                # NEW: count dropped frames for OOM stress test proof (Hint #2)
                with state_lock:
                    dashboard_state["frames_dropped"] += 1

        time.sleep(0.1)  # Maintain 10Hz frequency

# =================================================================
# 2. INFERENCE THREAD: EDGE AI ENGINE (Lab 9, 11, 13)
# =================================================================
def inference_worker():
    """
    Handles preprocessing and AI detection (YOLOv10-N / INT8).
    NEW: Power-Saving Mode — skips inference when scene is stable (Topic II hint).
    """
    print("[Inference] Thread started (Simulation Mode + Power-Save)")

    last_vehicle_count = None  # NEW: for power-saving comparison

    while not stop_event.is_set():
        try:
            data = video_queue.get(timeout=1)
            ts, frame = data["ts"], data["frame"]

            # --- Pre-processing (Lab 9/10 Vectorization) — original preserved ---
            resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)

            # --- AI Inference Simulation (Lab 11) ---
            vehicle_count = random.randint(2, 12)

            # NEW: Power-Saving Mode (Topic II Pro Hint adapted for Topic III)
            # Skip forwarding result if vehicle count is stable (delta < threshold)
            if last_vehicle_count is not None:
                delta = abs(vehicle_count - last_vehicle_count)
                if delta < POWER_SAVE_THRESHOLD:
                    with state_lock:
                        dashboard_state["power_save_skips"] += 1
                    last_vehicle_count = vehicle_count
                    continue  # suppress this inference result

            last_vehicle_count = vehicle_count

            with state_lock:
                dashboard_state["frames_processed"] += 1

            fused_data_queue.put({"ts": ts, "vehicle_count": vehicle_count})

        except queue.Empty:
            continue

# =================================================================
# 3. CONSUMER THREAD: FUSION & AGGREGATION (Lab 1, 4, 14)
# =================================================================
def data_manager_worker():
    """
    Executes Temporal Join, CSV Logging, and Cloud Exfiltration.
    NEW: Latency jitter profiling per record + anomaly flag in summary.
    """
    print("[Data Manager] Thread started (Aggregation + Jitter Profiling)")
    agg_buffer = []
    last_agg_timestamp = time.time()

    while not stop_event.is_set():
        try:
            inf_res = fused_data_queue.get(timeout=1)
            inf_ts = inf_res["ts"]

            # --- PART A: Heterogeneous Fusion (Lab 14) — original preserved ---
            best_match = None
            min_drift = float('inf')

            for s in list(sensor_buffer):
                drift = abs(inf_ts - s["ts"])
                if drift < min_drift and drift <= SYNC_TOLERANCE:
                    min_drift = drift
                    best_match = s

            if best_match:
                # NEW: Log sync jitter per record (Hint #2: Latency Jitter Profiling)
                sync_error_ms = min_drift * 1000
                with state_lock:
                    dashboard_state["sync_error_samples"].append(min_drift)
                    if len(dashboard_state["sync_error_samples"]) > 500:
                        dashboard_state["sync_error_samples"].pop(0)

                # Log to latency CSV
                with open(LATENCY_LOG_CSV, "a") as f:
                    f.write(
                        f"{datetime.fromtimestamp(inf_ts).strftime('%H:%M:%S.%f')[:-3]},"
                        f"{sync_error_ms:.3f},"
                        f"{inf_res['vehicle_count']},"
                        f"{best_match['value']},"
                        f"{'WITHIN_TOLERANCE' if sync_error_ms <= 100 else 'EXCEEDED'}\n"
                    )

                record = {
                    "ts": inf_ts,
                    "pm25": best_match["value"],
                    "vehicle_count": inf_res["vehicle_count"],
                    "sync_error_ms": round(sync_error_ms, 3)  # NEW field
                }
                agg_buffer.append(record)

                # --- PART B: Sliding Window Aggregation (Topic III) — original preserved ---
                if time.time() - last_agg_timestamp >= WINDOW_SIZE:
                    if agg_buffer:
                        avg_pm = sum(r["pm25"] for r in agg_buffer) / len(agg_buffer)
                        avg_vc = sum(r["vehicle_count"] for r in agg_buffer) / len(agg_buffer)
                        avg_jitter = sum(r["sync_error_ms"] for r in agg_buffer) / len(agg_buffer)

                        # NEW: Correlation flag — high traffic + high pollution
                        correlation_flag = (
                            "CONGESTION_POLLUTION_CORR"
                            if avg_vc > 8 and avg_pm > 30
                            else "NORMAL"
                        )

                        summary_payload = {
                            "summary_start": datetime.fromtimestamp(last_agg_timestamp).strftime('%H:%M:%S'),
                            "avg_pm25": round(avg_pm, 2),
                            "avg_vehicles": round(avg_vc, 1),
                            "samples": len(agg_buffer),
                            "avg_sync_jitter_ms": round(avg_jitter, 3),  # NEW
                            "alert": "HIGH_POLLUTION" if avg_pm > 35 else "NORMAL",
                            "correlation": correlation_flag,              # NEW
                            "anomaly_count": sum(                         # NEW
                                1 for r in agg_buffer if r["pm25"] > 50
                            )
                        }

                        with state_lock:
                            dashboard_state["last_summary"] = summary_payload

                        # --- PART C: Logging to CSV & Cloud (Lab 1 & 4) ---
                        transmit_and_log(summary_payload)

                        agg_buffer = []
                        last_agg_timestamp = time.time()

        except queue.Empty:
            continue

def transmit_and_log(payload):
    """
    Logs to CSV for Excel and handles Cloud transmission with Fallback.
    NEW: includes anomaly_count and correlation fields.
    """
    # 1. Log to CSV for Excel Analysis (Lab 1) — extended with new fields
    with open(PERF_LOG_CSV, "a") as f:
        line = (
            f"{payload['summary_start']},"
            f"{payload['avg_pm25']},"
            f"{payload['avg_vehicles']},"
            f"{payload['samples']},"
            f"{payload['alert']},"
            f"{payload.get('anomaly_count', 0)}\n"  # NEW column
        )
        f.write(line)

    # 2. MQTT with Local Fallback (Lab 4/14) — original logic preserved
    is_network_up = random.random() > 0.2

    if is_network_up:
        print(f"[CLOUD] ✅ MQTT Sent  | PM2.5={payload['avg_pm25']} | "
              f"Vehicles={payload['avg_vehicles']} | "
              f"Jitter={payload.get('avg_sync_jitter_ms','?')}ms | "
              f"Corr={payload.get('correlation','?')}")
        with state_lock:
            dashboard_state["cloud_sent"] += 1
    else:
        with open(FALLBACK_FILE, "a") as f:
            f.write(json.dumps(payload) + "\n")
        print(f"[LOCAL] ⚠️  Network Down — cached to {FALLBACK_FILE}")
        with state_lock:
            dashboard_state["fallback_cached"] += 1

# =================================================================
# MAIN EXECUTION
# =================================================================
if __name__ == "__main__":
    if not os.path.exists(VIDEO_SOURCE):
        print(f"[Error] Missing source video: {VIDEO_SOURCE}")
        print("[Info]  Place a video file named 'sample_video.mp4' in this directory.")
    else:
        producer  = threading.Thread(target=producer_worker, args=(VIDEO_SOURCE,), daemon=True)
        inference = threading.Thread(target=inference_worker, daemon=True)
        manager   = threading.Thread(target=data_manager_worker, daemon=True)
        dashboard = threading.Thread(target=print_dashboard, daemon=True)  # NEW

        print("\n" + "="*60)
        print("  🚀  Edge AI Gateway — Urban Mobility & Emission Monitor")
        print("="*60)
        print(f"  Topic     : III — Urban Mobility & Emission Correlation")
        print(f"  Sensors   : PM2.5 @ 10Hz (scalar) + Vehicle count (tensor)")
        print(f"  Window    : {WINDOW_SIZE}s sliding aggregation")
        print(f"  Tolerance : {int(SYNC_TOLERANCE*1000)}ms sync threshold")
        print(f"  PowerSave : skip inference if Δvehicles < {POWER_SAVE_THRESHOLD}")
        print(f"  Outputs   : {PERF_LOG_CSV}, {LATENCY_LOG_CSV}, {FALLBACK_FILE}")
        print("="*60 + "\n")

        producer.start()
        inference.start()
        manager.start()
        dashboard.start()

        print(f"[Main] Pipeline akan berhenti otomatis dalam {RUN_DURATION} detik...\n")

        try:
            start_time = time.time()
            while time.time() - start_time < RUN_DURATION:
                elapsed = int(time.time() - start_time)
                remaining = RUN_DURATION - elapsed
                # Tampilkan countdown setiap 30 detik
                if elapsed % 30 == 0 and elapsed > 0:
                    print(f"[Main] ⏱  {elapsed}s elapsed — {remaining}s remaining until auto-stop...")
                time.sleep(1)

            print(f"\n[Main] ✅ {RUN_DURATION}s selesai — menghentikan semua thread...")

        except KeyboardInterrupt:
            print("\n[Main] Shutdown manual diterima...")

        finally:
            stop_event.set()
            producer.join()
            inference.join()
            manager.join()

            # ── FINAL SUMMARY ──────────────────────────────────────
            with state_lock:
                s = dashboard_state
                avg_j = (
                    sum(s["sync_error_samples"]) / len(s["sync_error_samples"]) * 1000
                    if s["sync_error_samples"] else 0
                )
            print("\n" + "="*60)
            print("  🏁  FINAL REPORT — Edge AI Gateway")
            print("="*60)
            print(f"  Total Frames Processed : {s['frames_processed']}")
            print(f"  Total Frames Dropped   : {s['frames_dropped']}")
            print(f"  Power-Save Skips       : {s['power_save_skips']}")
            print(f"  Avg Sync Jitter        : {avg_j:.2f} ms")
            print(f"  Anomalies Detected     : {s['anomalies_injected']}")
            print(f"  Cloud MQTT Sent        : {s['cloud_sent']}")
            print(f"  Local Fallback Cached  : {s['fallback_cached']}")
            print(f"\n  Output files:")
            print(f"    📄 {PERF_LOG_CSV}")
            print(f"    📄 {LATENCY_LOG_CSV}")
            print(f"    📄 {FALLBACK_FILE}")
            print("="*60)
            print("[Main] Pipeline selesai.")