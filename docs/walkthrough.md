# Project Aegis - Complete Technical Walkthrough

Welcome to the final technical summary for **Project Aegis**, an edge-AI security testbed built to detect, attack, and defend power grid anomalies. We have successfully completed all four phases of the project!

## 🔹 Phase 1: Setup & Data Ingestion
- **Environment:** Created a hardware-accelerated Conda environment (`project_aegis`).
- **Data Acquisition:** Downloaded the **HAI Security Dataset** (version 23.05) from Kaggle.
- **Pipeline Setup:** Established the repository architecture (`data/`, `models/`, `src/`, `docs/`, `scripts/`).

## 🔹 Phase 2: Anomaly Detection Architecture
- **Preprocessing:** Built a PyTorch DataLoader (`dataset.py`) that normalizes 86 telemetry features using `MinMaxScaler` and generates 60-second sliding windows to capture temporal dynamics.
- **Model:** Developed a **1D-CNN Autoencoder** (`autoencoder.py`) optimized for edge devices.
- **Results:** 
    - Trained on ~1.1 million data points.
    - Evaluated on ~284,000 test windows.
    - Automatically flagged **14,217** anomalous windows by calculating a 95th percentile reconstruction error threshold.

## 🔹 Phase 3: Red & Blue Teaming (The "Robustness Paradox")
- **Red Team (Attack):** Implemented a custom **FGSM Anomaly Hiding Attack** (`fgsm_attack.py`). By introducing minor adversarial noise, the attack successfully forced the autoencoder to perfectly reconstruct anomalous data, hiding **90.16%** of anomalies from the detector!
- **Blue Team (Defense):** Hardened the model using **Adversarial Training** (`adv_train.py`). We augmented the training loop to generate adversarial "False Alarm" noise and trained the model to ignore it.
- **The Paradox:** When we tested the hardened model against the Anomaly Hiding attack, the attack success rate jumped to **100%**. Defending against one objective (False Alarms) inadvertently made the model highly vulnerable to the opposite objective (Anomaly Hiding).

## 🔹 Phase 4: Advanced Defenses, Visualization, & Deployment
We expanded the project to make it production-ready and visually accessible:

### 1. Ensemble Defense (`ensemble.py`)
To mitigate the vulnerabilities discovered in Phase 3, we built an **Ensemble Autoencoder**. By training three separate models and averaging their outputs, we significantly increased the difficulty of the attack. The FGSM attack success rate immediately dropped from 90% down to **61.5%**.

### 2. Visualization (`visualize.py`)
We created a `matplotlib` script to visualize the cyber-attacks. It extracts a genuine anomaly, generates the adversarial noise, and plots a side-by-side comparison of:
- The original anomaly
- The adversarial perturbation
- The poisoned data
You can view the generated chart at `docs/anomaly_visualization.png`.

### 3. Containerization
To simulate an Edge AI deployment, we containerized the entire pipeline.
- We extracted the core dependencies into a lean `requirements.txt`.
- We wrote a `Dockerfile` based on `python:3.10-slim`.
- We configured a `docker-compose.yml` to automatically launch the container and run the evaluation pipeline, proving the project can be deployed seamlessly to any hardware.

---

**Thank you for exploring Project Aegis!** This project successfully demonstrates end-to-end expertise in AI engineering, adversarial machine learning, and cybersecurity testbed deployment.
