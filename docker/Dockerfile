# FU_Biometry private-test submission — the 22.95 transferable ensemble.
# Fully offline: all 5 model weights are baked in; no runtime downloads.
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

WORKDIR /app

# Force offline: nothing may touch the network at runtime (--network none).
ENV HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    HF_DATASETS_OFFLINE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# System deps for OpenCV / albumentations
RUN apt-get update -qq && apt-get install -y -qq --no-install-recommends \
    libgl1-mesa-glx libglib2.0-0 libsm6 libxrender1 libxext6 \
    && rm -rf /var/lib/apt/lists/*

# Python deps (pinned to the validated stack)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Entry script (organizer-provided, unmodified) + inference code
COPY predict.py model.py /app/

# Vendored, validated source (factories, transforms with pinned OURS stats, DARK decoder, USFM graft)
COPY vendor/ /app/vendor/

# All five model weights — baked into the image (offline)
COPY weights/ /app/weights/

# Placeholder required by predict.py's copy step (real weights live in /app/weights/)
COPY best_model.pth /app/best_model.pth

ENTRYPOINT ["python3", "/app/predict.py"]
