FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    YOLO_DEVICE=cpu

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-runtime.txt /app/requirements-runtime.txt
RUN python -m pip install --upgrade pip && \
    pip install --no-cache-dir --prefer-binary --only-binary=:all: -r /app/requirements-runtime.txt && \
    rm -rf /root/.cache/pip

COPY merge_ballot_logs.py /app/merge_ballot_logs.py
COPY runtime_pipeline.py /app/runtime_pipeline.py
COPY remove-background /app/remove-background
COPY run_full_extraction.py /app/run_full_extraction.py
COPY upload_server.py /app/upload_server.py
COPY lambda_handler.py /app/lambda_handler.py
COPY current_box.json /app/current_box.json

COPY stamp-detection/src /app/stamp-detection/src
COPY stamp-detection/configs /app/stamp-detection/configs
COPY stamp-detection/runs/train/yolo_stamp/weights/best.pt /app/stamp-detection/runs/train/yolo_stamp/weights/best.pt

COPY Handwritten-Digit-Recognition/ballot_reader /app/Handwritten-Digit-Recognition/ballot_reader
COPY Handwritten-Digit-Recognition/tf-cnn-model.keras /app/Handwritten-Digit-Recognition/tf-cnn-model.keras

RUN mkdir -p /app/runtime/logs /app/runtime/debug_ballot /app/runtime/stamp_outputs /app/uploads && \
    useradd --create-home --shell /bin/bash appuser && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/healthz')"

CMD ["python", "upload_server.py"]
