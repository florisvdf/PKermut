FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    gcc \
    g++ \
    wget \
    libopenmpi-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ml/code

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

COPY pyproject.toml .
COPY uv.lock .
COPY README.md .
COPY kermut kermut
COPY pg_model pg_model

RUN uv sync # Can't do no install trick because namespace package depends on workspace members. no-install-project will therefore install pg-model and kermut

ENV PYTHONPATH="${PYTHONPATH}:/opt/ml/code"

ENTRYPOINT ["uv", "run", "python", "/opt/ml/code/pg_model/src/pg_model/__main__.py", "train"]
