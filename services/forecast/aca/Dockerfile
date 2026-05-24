# Minimal image for distributed SARIMA fan-out on Azure Container Apps.
#
# Strategy: pin the dependency set to the slim path (numpy + pandas + duckdb
# + statsmodels + the forecast package). Skip heavy backends (LightGBM,
# Prophet, Transformer/torch, MLflow) — workers only need to run SARIMA +
# the naive/mean/drift baselines. The driver assembles partials at the end.

FROM python:3.12-slim AS base

# System libs needed for statsmodels (scipy → lapack/blas), curl for healthcheck.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libopenblas0 liblapack3 libgomp1 ca-certificates curl tini \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Layer 1: deps (rebuilds rarely)
COPY services/forecast/aca/requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade pip wheel \
 && pip install --no-cache-dir -r /tmp/requirements.txt

# Layer 2: source (rebuilds on every commit)
COPY services /app/services
COPY pyproject.toml /app/pyproject.toml

# Confirm import surface at build time so deploy doesn't silently ship broken code.
RUN python -c "from services.forecast.aca_worker import main; print('worker import OK')"

# Use tini as PID 1 so SIGTERM is forwarded correctly when ACA cancels the job.
ENTRYPOINT ["/usr/bin/tini", "--"]

# Default CMD is the worker. ACA injects PARTITION_ID, TOTAL_PARTITIONS, etc.
CMD ["python", "-m", "services.forecast.aca_worker", "-v"]
