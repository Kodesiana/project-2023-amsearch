FROM python:3.11-slim-bookworm AS base

WORKDIR /usr/app

COPY requirements.txt .

RUN apt-get update \
    && apt-get -y install libpq-dev gcc \
    && pip install --no-cache-dir -r requirements.txt

COPY . ./amsearch
COPY ./logging.conf .

EXPOSE 8000

CMD ["gunicorn", "--log-config", "logging.conf", "--workers", "2", "--forwarded-allow-ips", "*", "--bind", "0.0.0.0:8000", "amsearch:app"]
