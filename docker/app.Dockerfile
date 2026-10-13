FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /workspace
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 curl && rm -rf /var/lib/apt/lists/*
RUN pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
COPY app/requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
COPY app /workspace/app
COPY scripts /workspace/scripts
COPY knowledge_base /workspace/knowledge_base
COPY docker/app-entrypoint.sh /usr/local/bin/app-entrypoint.sh
RUN chmod +x /usr/local/bin/app-entrypoint.sh
EXPOSE 8501
HEALTHCHECK CMD curl -fs http://localhost:8501/_stcore/health || exit 1
ENTRYPOINT ["app-entrypoint.sh"]
