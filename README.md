# Project Aegis

**Project Aegis** is an edge AI security testbed focused on power grid anomaly detection. This repository provides a framework to simulate, attack, and defend an AI-driven edge sensor in an Industrial Control System (ICS) environment, aligning with the core goals of the DOE Genesis Mission.

## Architecture Overview

- **`data/`**: Raw and processed telemetry datasets.
- **`src/model/`**: PyTorch implementation of a lightweight power grid anomaly detection model.
- **`src/edge_sim/`**: Edge environment simulation, mocking device inference and protocol communications (e.g., MQTT/Modbus).
- **`src/attack/`**: "Red Team" scripts for adversarial attacks (FGSM, PGD) utilizing the Adversarial Robustness Toolbox.
- **`src/defense/`**: "Blue Team" hardening tools (adversarial training, input sanitization).
- **`docs/`**: Threat models and vulnerability reports.

## Setup

1. Create the Conda environment and install dependencies:
   ```bash
   conda env create -f environment.yml
   conda activate project_aegis
   ```
2. Download datasets:
   ```bash
   python3 scripts/download_data.py
   ```