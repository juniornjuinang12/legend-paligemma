FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/data/hf \
    HF_HUB_CACHE=/data/hf/hub \
    HF_DATASETS_CACHE=/data/hf/datasets \
    TRANSFORMERS_CACHE=/data/hf/transformers

WORKDIR /app
RUN mkdir -p /data /output

RUN pip install --no-cache-dir transformers accelerate pillow huggingface_hub safetensors

COPY main.py /app/main.py

CMD ["torchrun", "--standalone", "--nproc_per_node=8", "/app/main.py"]
