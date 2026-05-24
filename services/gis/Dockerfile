FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir duckdb==1.5.2 requests==2.32.3 azure-storage-blob==12.22.0

COPY services/powerbi/fabric_publish.py /app/services/powerbi/fabric_publish.py
COPY services/__init__.py /app/services/__init__.py
COPY services/powerbi/__init__.py /app/services/powerbi/__init__.py
COPY services/gis/__init__.py /app/services/gis/__init__.py
COPY services/gis/aggregate_state_rx.py /app/services/gis/aggregate_state_rx.py
COPY services/gis/run_job.py /app/services/gis/run_job.py

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["python3", "/app/services/gis/run_job.py"]
