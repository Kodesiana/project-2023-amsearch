# pull base image
FROM python:3.11-bookworm AS base

# set work directory
WORKDIR /usr/app

# install system dependencies
RUN apt-get update \
    && apt-get -y install libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple

# create prod image
FROM base AS prod

# copy project
COPY . ./amsearch
COPY ./logging.conf .

# expose port
EXPOSE 8000

# run server
CMD ["gunicorn", "--log-config", "logging.conf", "--workers", "1", "--forwarded-allow-ips", "*", "--bind", "0.0.0.0:8000", "amsearch:app"]
