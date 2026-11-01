# Inference service image (Task 3). CPU-only, pinned, and WITHOUT model weights.
# The default PyTorch wheels bundle CUDA/GPU libraries you do not need; installing the CPU
# wheel keeps the image well under the 2 GB limit.
FROM python:3.11-slim

WORKDIR /app

# CPU-only torch first (pinned), from the PyTorch CPU index.
RUN pip install --no-cache-dir torch==2.4.1 --index-url https://download.pytorch.org/whl/cpu

COPY service/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY service/app.py service/model_runner.py ./
COPY data/schema.json ./schema.json

ENV PYTHONUNBUFFERED=1 MAX_NEW_TOKENS=64
EXPOSE 8000

# Readiness/liveness are handled by Kubernetes probes; one worker is enough per replica.
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
