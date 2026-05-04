FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Dependencias de sistema necessarias para algumas libs Python
# (build-essential e curl ajudam em wheels que precisam compilar)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       build-essential curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copia o codigo necessario para a instalacao do pacote.
# (`pip install .` requer o pacote `src/` presente para ler metadados.)
COPY pyproject.toml ./
COPY src/ ./src/

# Instala o pacote em modo runtime (sem extras de dev) + modelos spaCy
# (necessarios pelo Presidio para detectar PII em PT/EN).
RUN pip install --no-cache-dir . \
    && python -m spacy download en_core_web_sm \
    && python -m spacy download pt_core_news_sm

COPY configs/ ./configs/
COPY models/ ./models/

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
