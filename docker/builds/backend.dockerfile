FROM python:3.13.5-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir fastapi uvicorn transformers accelerate bitsandbytes prometheus-client

WORKDIR /app

COPY src/backend.py /app/src/

EXPOSE 8000

CMD ["uvicorn", "src.backend:song_app", "--host", "0.0.0.0", "--port", "8000"]