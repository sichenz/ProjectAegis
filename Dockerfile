FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
COPY src/ /app/src/
COPY scripts/ /app/scripts/
COPY models/ /app/models/
COPY data/ /app/data/

# By default, run the evaluation script to prove the container works
CMD ["python", "src/model/evaluate.py"]
