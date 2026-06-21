FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y \
    python3 python3-pip python3-dev \
    git wget curl \
    libglib2.0-0 libgl1 libgomp1 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip setuptools wheel

RUN python3 -m pip install paddlepaddle-gpu==3.2.1 \
    -i https://www.paddlepaddle.org.cn/packages/stable/cu126/

RUN python3 -m pip install -U "paddleocr[doc-parser]>=3.6.0"

COPY requirements.txt .
RUN python3 -m pip install -r requirements.txt

COPY handler.py .

CMD ["python3", "-u", "handler.py"]