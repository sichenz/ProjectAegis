FROM python:3.10-slim

LABEL maintainer="Project Aegis Team"
LABEL description="Edge AI security testbed for ICS anomaly detection"

WORKDIR /app

# Install dependencies first (cache-friendly layer ordering)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ /app/src/
COPY scripts/ /app/scripts/

# Copy trained models
COPY models/ /app/models/

# Copy only the dataset version we use
COPY data/hai-23.05/ /app/data/hai-23.05/

# Create a non-root user for security
RUN useradd --create-home appuser
USER appuser

# By default, run the evaluation script to prove the container works
CMD ["python", "src/model/evaluate.py"]
