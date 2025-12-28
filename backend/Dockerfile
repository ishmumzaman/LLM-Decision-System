# syntax=docker/dockerfile:1

FROM python:3.11-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# faiss-cpu wheels typically rely on OpenMP runtime
RUN apt-get update \
  && apt-get install -y --no-install-recommends libgomp1 \
  && rm -rf /var/lib/apt/lists/*

# Install deps first (better layer caching)
COPY backend/requirements.txt /app/backend/requirements.txt
RUN python -m pip install --upgrade pip \
  && python -m pip install -r /app/backend/requirements.txt

# Copy app code + domain artifacts/config (artifacts are committed; corpora are ignored by .dockerignore)
COPY backend/ /app/backend/
COPY domains/ /app/domains/

EXPOSE 8000

CMD ["python", "-m", "uvicorn", "app.main:app", "--app-dir", "backend", "--host", "0.0.0.0", "--port", "8000"]

