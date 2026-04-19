# Project Aegis

**Project Aegis** is an edge AI security testbed focused on power grid anomaly detection. This repository provides a complete framework to simulate, attack, and defend an AI-driven edge sensor in an Industrial Control System (ICS) environment, aligning with the core goals of the DOE Genesis Mission.

## Key Results

| Metric | Value |
|--------|-------|
| Training Data Points | ~1.1 million |
| Test Windows Evaluated | 284,341 |
| Anomalies Detected (Baseline) | 14,217 (5%) |
| FGSM Attack Success vs. Baseline | 90.16% |
| FGSM Attack Success vs. Robust Model | 100% (Robustness Paradox) |
| FGSM Attack Success vs. Ensemble | 61.5% |

## Architecture Overview

```
ProjectAegis/
├── data/                  # Raw HAI Security Dataset (downloaded via Kaggle)
├── models/                # Trained model weights (.pth) and scaler (.pkl)
├── src/
│   ├── model/
│   │   ├── autoencoder.py # 1D-CNN Autoencoder for anomaly detection
│   │   ├── dataset.py     # PyTorch DataLoader with MinMaxScaler + sliding windows
│   │   ├── train.py       # Training pipeline
│   │   └── evaluate.py    # Evaluation with 95th-percentile thresholding
│   ├── attack/
│   │   └── fgsm_attack.py # FGSM adversarial attack (anomaly hiding)
│   ├── defense/
│   │   ├── adv_train.py   # Adversarial training (Blue Team hardening)
│   │   └── ensemble.py    # Ensemble Autoencoder defense
│   └── edge_sim/          # Edge environment simulation (future work)
├── scripts/
│   ├── download_data.py   # Kaggle dataset downloader
│   └── visualize.py       # Matplotlib attack visualization
├── docs/                  # Walkthrough and visualizations
├── Dockerfile             # Containerized edge deployment
├── docker-compose.yml     # Orchestration config
├── environment.yml        # Conda environment specification
└── requirements.txt       # Pip dependencies (for Docker)
```

## Setup

### Option 1: Conda (Recommended for Development)

```bash
# Create and activate the environment
conda env create -f environment.yml
conda activate project_aegis

# Download the HAI Security Dataset (requires Kaggle API key)
python scripts/download_data.py
```

### Option 2: Docker (Recommended for Deployment)

```bash
# Build and run the evaluation pipeline
docker-compose up --build

# Or run a specific script
docker-compose run edge-ai-node python src/attack/fgsm_attack.py --model ensemble.pth
```

## Usage

### Phase 2: Train and Evaluate the Anomaly Detector

```bash
# Train the 1D-CNN Autoencoder on normal telemetry data
python src/model/train.py

# Evaluate on test data and detect anomalies
python src/model/evaluate.py
```

### Phase 3: Red & Blue Teaming

```bash
# Run FGSM attack against the baseline model
python src/attack/fgsm_attack.py

# Run adversarial training to harden the model
python src/defense/adv_train.py

# Test the attack against the hardened model
python src/attack/fgsm_attack.py --model autoencoder_robust.pth
```

### Phase 4: Advanced Defense & Visualization

```bash
# Train the ensemble defense
python src/defense/ensemble.py

# Test the attack against the ensemble
python src/attack/fgsm_attack.py --model ensemble.pth

# Generate attack visualization
python scripts/visualize.py
```

## Dataset

This project uses the **HAI Security Dataset** (v23.05), a public ICS cybersecurity dataset from the HIL-based Augmented ICS testbed, containing 86 telemetry features from a simulated power grid. Download it using `scripts/download_data.py` with a valid Kaggle API key.

## Technical Details

For a complete technical walkthrough of all four project phases, including methodology, results, and analysis, see [docs/walkthrough.md](docs/walkthrough.md).