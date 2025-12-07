# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Dépendances minimales
RUN pip install --no-cache-dir --upgrade pip \
 && pip install --no-cache-dir huggingface_hub

# Copier ton script
COPY main.py /app/main.py

# Dossier de data (monté en volume sur SaladCloud ou local)
RUN mkdir -p /data

# Lancer le script
ENTRYPOINT ["python", "/app/main.py"]
