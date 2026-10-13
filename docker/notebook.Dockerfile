FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PIP_NO_CACHE_DIR=1
RUN apt-get update && apt-get install -y --no-install-recommends build-essential libgomp1 git && rm -rf /var/lib/apt/lists/*
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY notebooks/requirements-notebook.txt /tmp/requirements-notebook.txt
RUN pip install -r /tmp/requirements-notebook.txt
WORKDIR /workspace
EXPOSE 8888
