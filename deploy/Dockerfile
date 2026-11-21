FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PT_DATA_DIR=/data
WORKDIR /srv/pt
COPY deploy/requirements.txt /tmp/pt-requirements.txt
RUN pip install --no-cache-dir -r /tmp/pt-requirements.txt \
    && groupadd --gid 10001 pt && useradd --uid 10001 --gid 10001 pt \
    && install -d -o 10001 -g 10001 -m 0700 /data
COPY app/__init__.py app/__init__.py
COPY app/operator app/operator
COPY noomo-ui noomo-ui
USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "app.operator.web:from_environment", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
