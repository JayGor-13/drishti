# DRISHTI training image for the DGX.
# Build (from the repo root on the DGX):  docker build -t drishti:latest -f dgx/Dockerfile dgx
# If Docker Hub is blocked on the DGX, use the NVIDIA base instead:
#   FROM nvcr.io/nvidia/pytorch:24.10-py3
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

RUN pip install --no-cache-dir numpy pillow pyyaml tqdm matplotlib psutil pytest

ENV PYTHONUNBUFFERED=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/mpl \
    HOME=/tmp
WORKDIR /workspace/code
