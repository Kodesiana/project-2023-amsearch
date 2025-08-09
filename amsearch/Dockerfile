# pull base image
FROM python:3.13-bookworm AS builder

# install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# set env
ENV UV_PYTHON_DOWNLOADS=0
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# set work directory
WORKDIR /app/amsearch

# install python dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --locked --no-install-project --no-dev

# copy project
COPY . /app/amsearch

# copy cache
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

FROM python:3.13-slim-bookworm AS prod

# set workdir
WORKDIR /app

# copy the application from the builder
COPY --from=builder --chown=app:app /app/amsearch /app/amsearch

# set path to venv
ENV PATH="/app/amsearch/.venv/bin:$PATH"

# expose port
EXPOSE 8000

# run server
CMD ["gunicorn", "--log-config", "amsearch/logging.conf", "--workers", "1", "--forwarded-allow-ips", "*", "--bind", "0.0.0.0:8000", "amsearch:app"]
