FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data/processed /app/data/monitoring /app/models /app/reports && \
    chown -R appuser:appuser /app

COPY --chown=appuser:appuser app.py .
COPY --chown=appuser:appuser model.pkl .
COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser models/ models/
COPY --chown=appuser:appuser reports/ reports/
COPY --chown=appuser:appuser data/processed/train.csv data/processed/train.csv

USER appuser
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0", "--server.headless=true"]