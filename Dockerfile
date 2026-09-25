FROM python:3.11-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    DATA_DIR=/app/data \
    ARTIFACTS_DIR=/app/artifacts

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY hostembed ./hostembed
COPY app ./app
COPY scripts ./scripts
COPY docker-entrypoint.sh .
RUN chmod +x docker-entrypoint.sh

EXPOSE 8000

# При старте: если артефактов нет — сгенерировать данные и прогнать пайплайн,
# затем поднять веб-сервис.
ENTRYPOINT ["./docker-entrypoint.sh"]
