FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf \
    HF_HUB_CACHE=/data/hf/hub \
    HF_DATASETS_CACHE=/data/hf/datasets \
    TRANSFORMERS_CACHE=/data/hf/transformers

WORKDIR /app

# Dossiers de cache HF + sortie
RUN mkdir -p /data/hf/hub /data/hf/datasets /data/hf/transformers /output

RUN pip install --no-cache-dir transformers accelerate pillow huggingface_hub safetensors

COPY main.py /app/main.py

# IMPORTANT: version "test logs" (simple, fiable sur Salad)
CMD ["python", "-u", "/app/main.py"]
