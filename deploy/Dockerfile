FROM docker.io/library/python:3.12-slim@sha256:05a994d2344eb364cc186910f4fc8786a38215f39f00d0acc25bff33a08f4177
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PT_DATA_DIR=/data
WORKDIR /srv/pt
COPY deploy/requirements.txt deploy/requirements.lock /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt -c /tmp/requirements.lock \
    && groupadd --gid 10001 pt && useradd --uid 10001 --gid 10001 pt \
    && install -d -o 10001 -g 10001 -m 0700 /data
COPY app/__init__.py app/__init__.py
COPY app/operator app/operator
COPY noomo-ui noomo-ui
COPY deploy/alembic.ini deploy/alembic.ini
COPY deploy/alembic deploy/alembic
COPY deploy/export-static.py deploy/export-static.py
RUN chmod -R a+rX /srv/pt
USER 10001:10001
EXPOSE 8000
CMD ["uvicorn", "app.operator.web:from_environment", "--factory", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log", "--no-proxy-headers"]
