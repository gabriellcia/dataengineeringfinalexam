# End-to-End Edge AI Pipeline: Urban Mobility & Emission Correlation
**Capstone Project - Data Engineering CSIE Tamkang University**

## 📌 Project Overview
[cite_start]Proyek ini mengimplementasikan sistem **Edge AI Gateway** industri untuk mengorelasikan kepadatan lalu lintas dengan kualitas udara secara real-time[cite: 91, 93]. [cite_start]Sistem dirancang untuk berjalan di lingkungan terbatas (2-Core CPU) dengan fokus pada stabilitas arsitektur dan efisiensi data engineering[cite: 23, 25].

## 🛠 Key Features (W1-W14 Integration)
* [cite_start]**Multi-threaded Architecture**: Pemisahan Producer, Inference, dan Consumer menggunakan `queue.Queue(maxsize=1)` untuk mencegah kebocoran memori (OOM)[cite: 31, 56, 57].
* [cite_start]**Heterogeneous Data Fusion**: Sinkronisasi *Temporal Join* antara data sensor PM2.5 (10Hz) dan deteksi kendaraan YOLO (10Hz) dengan toleransi <100ms[cite: 37, 59, 95, 96].
* [cite_start]**Quantized Edge Inference**: Menggunakan model **TFLite INT8** untuk efisiensi pemrosesan pada CPU terbatas[cite: 34].
* [cite_start]**Fault Tolerance & Resilience**: Mekanisme *Exponential Backoff* untuk pengiriman MQTT dan **Local Fallback (JSONL)** jika terjadi kegagalan jaringan[cite: 40, 61, 62].
* [cite_start]**Edge Analytics**: Implementasi *Sliding Window Aggregation* 30 detik untuk mengirimkan ringkasan data ke cloud guna menghemat bandwidth[cite: 97].

## 🏗 System Architecture
1.  [cite_start]**Producer**: Simulasi kamera (Virtual Camera .mp4) dan sensor IoT (Mathematical Mocking)[cite: 47, 49].
2.  [cite_start]**Inference**: Pre-processing gambar ter-vektorisasi (NumPy) dan deteksi objek[cite: 71].
3.  [cite_start]**Data Manager**: Melakukan join data, agregasi, dan manajemen pengiriman data ke broker MQTT[cite: 132].

## 🚀 Setup & Installation
1.  **Clone Repository**:
    ```bash
    git clone [https://github.com/username/capstone-edge-ai.git](https://github.com/username/capstone-edge-ai.git)
    cd capstone-edge-ai
    ```
2.  **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
3.  **Run Pipeline**:
    ```bash
    python main.py
    ```

## 📊 Performance Metrics
* [cite_start]**Inference Latency**: Dioptimalkan menggunakan model INT8[cite: 138].
* [cite_start]**Sync Error**: Dijamin dalam rentang <100ms melalui *Nearest-Neighbor sync*[cite: 59].
* [cite_start]**Reliability**: Teruji melalui mekanisme *Dead Letter Queue* (DLQ) untuk data yang gagal terkirim[cite: 61].

## 📝 Submission Details
* [cite_start]**Pitch Structure**: 5-8 Menit[cite: 139].
* [cite_start]**Video Demo**: Rekaman terminal & dashboard (1-2 menit)[cite: 116].
* [cite_start]**Deadline**: Slides & GitHub link dikirim 24 jam sebelum jadwal presentasi.