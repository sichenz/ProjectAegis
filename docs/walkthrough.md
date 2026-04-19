# Project Aegis — Complete Technical Walkthrough

Welcome to the final technical summary for **Project Aegis**, an edge-AI security testbed built to detect, attack, and defend power grid anomalies. This document covers all four completed phases of the project.

---

## 🔹 Phase 1: Setup & Data Ingestion
- **Environment:** Created a hardware-accelerated Conda environment (`project_aegis`) with PyTorch, scikit-learn, pandas, matplotlib, and the Adversarial Robustness Toolbox.
- **Data Acquisition:** Downloaded the **HAI Security Dataset** (version 23.05) from Kaggle — a public ICS cybersecurity dataset from the HIL-based Augmented ICS testbed, containing 86 telemetry features across multiple operational scenarios.
- **Pipeline Setup:** Established the repository architecture (`data/`, `models/`, `src/`, `docs/`, `scripts/`) with proper Python package structure (`__init__.py` files throughout).

---

## 🔹 Phase 2: Anomaly Detection Architecture
- **Preprocessing (`dataset.py`):** Built a PyTorch `Dataset` and `DataLoader` that normalizes all 86 telemetry features using `MinMaxScaler` and generates 60-second sliding windows to capture temporal dynamics in the sensor data.
- **Model (`autoencoder.py`):** Developed a **1D-CNN Autoencoder** optimized for edge devices:
    - **Encoder:** 3 convolutional layers (86 → 32 → 16 → 8 channels) with stride-2 downsampling
    - **Decoder:** 3 transposed convolutional layers mirroring the encoder with Sigmoid output
    - Input/output shape: `(batch, 60, 86)` — fully reconstructing the input window
- **Training (`train.py`):** Trained for 5 epochs on ~1.1 million data points using MSE loss and Adam optimizer, achieving a final loss of `0.005015`.
- **Evaluation (`evaluate.py`):** Evaluated on ~284,000 test windows. Used the 95th percentile of reconstruction errors as the anomaly threshold (`0.016369`), automatically flagging **14,217** anomalous windows (5%).

---

## 🔹 Phase 3: Red & Blue Teaming

### Red Team — FGSM Anomaly Hiding Attack (`fgsm_attack.py`)
We implemented a custom **Fast Gradient Sign Method (FGSM)** attack tailored for autoencoders. Unlike standard FGSM (which maximizes loss), our variant *minimizes* reconstruction error to hide anomalies:

```
perturbed = original - ε · sign(∇_x Loss)
```

With `ε = 0.05`, the attack successfully hid **90.16%** of anomalies from the detector by reducing their reconstruction error below the threshold.

### Blue Team — Adversarial Training (`adv_train.py`)
We hardened the model by fine-tuning it with adversarial examples. The training loop:
1. Generates adversarial "false alarm" noise (maximizing reconstruction error on clean data)
2. Trains the model to reconstruct the *original* clean data from both normal and adversarial inputs
3. Uses a 50/50 combined loss: `L = 0.5 · L_normal + 0.5 · L_adversarial`

### The Robustness Paradox
When we tested the hardened model against the anomaly-hiding attack, the attack success rate jumped to **100%**. By training the model to be robust against false alarm noise (high-error perturbations), we inadvertently made it better at reconstructing *any* perturbed input — including adversarially-hidden anomalies. This is a well-documented trade-off in adversarial ML literature, and it motivated the ensemble approach in Phase 4.

---

## 🔹 Phase 4: Advanced Defenses, Visualization & Deployment

### 1. Ensemble Defense (`ensemble.py`)
To mitigate the robustness paradox, we built an **Ensemble Autoencoder** containing three independently-initialized `Conv1DAutoencoder` instances, managed via `nn.ModuleList`. Each sub-model is trained sequentially on the same data but with different random initializations, creating natural diversity.

At inference time, the ensemble averages the reconstructions from all three models. An attacker must now craft a single perturbation that simultaneously fools all three — a significantly harder optimization problem.

**Result:** The FGSM attack success rate dropped from 90% to **61.5%**, demonstrating a meaningful improvement in robustness without the computational overhead of min-max optimization.

### 2. Visualization (`visualize.py`)
We created a `matplotlib` visualization script that:
1. Scans the test set to find a genuine anomaly (reconstruction error above threshold)
2. Applies the FGSM attack to generate adversarial noise
3. Plots three panels showing the top-3 most volatile sensors:
    - **Original anomaly** with its high reconstruction error
    - **Adversarial perturbation** (the noise added by the attacker)
    - **Poisoned data** with its now-reduced reconstruction error

The output chart is saved to `docs/anomaly_visualization.png`.

### 3. Containerization
To simulate an Edge AI deployment, we containerized the entire pipeline:
- **`requirements.txt`**: Extracted core pip dependencies with version pins for reproducible builds
- **`Dockerfile`**: Based on `python:3.10-slim`, with selective data copying (only `hai-23.05`), non-root user execution, and proper metadata labels
- **`docker-compose.yml`**: Orchestrates the container with volume mounts for persisting output back to the host
- **`.dockerignore`**: Excludes `.git/`, `__pycache__/`, unnecessary dataset versions, and documentation from the build context

---

## Summary

| Phase | Component | Key Outcome |
|-------|-----------|-------------|
| 1 | Setup | Conda environment + HAI dataset acquired |
| 2 | Detection | 1D-CNN Autoencoder flags 14,217 anomalies (5%) |
| 3 | Red Team | FGSM hides 90.16% of anomalies |
| 3 | Blue Team | Adversarial training reveals the Robustness Paradox |
| 4 | Ensemble | Attack success drops to 61.5% |
| 4 | Visualization | 3-panel attack comparison chart |
| 4 | Deployment | Dockerized edge-ready container |

**Thank you for exploring Project Aegis!** This project demonstrates end-to-end expertise in AI engineering, adversarial machine learning, and edge cybersecurity deployment.
