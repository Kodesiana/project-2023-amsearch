# pull base image
FROM pytorch/pytorch AS base

# set work directory
WORKDIR /usr/app

# install system dependencies
RUN apt-get update \
    && apt-get -y install libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# copy project
COPY . ./amsearch
COPY ./logging.conf .

# expose port
EXPOSE 8000

# run server
CMD ["gunicorn", "--log-config", "logging.conf", "--workers", "2", "--forwarded-allow-ips", "*", "--bind", "0.0.0.0:8000", "amsearch:app"]
