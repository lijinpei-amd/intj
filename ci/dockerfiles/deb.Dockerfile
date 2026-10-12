# CI image for apt-based cells; build args come from ci/image.py.
ARG BASE
FROM ${BASE}
LABEL org.opencontainers.image.source=https://github.com/lijinpei-amd/intj
ARG PY_PKGS
ARG PY_BIN
ARG TORCH_INDEX
ARG TRITON
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends ${PY_PKGS} gcc g++ git ca-certificates util-linux \
 && rm -rf /var/lib/apt/lists/*
COPY install-venv.sh /tmp/install-venv.sh
RUN sh /tmp/install-venv.sh
ENV PATH=/opt/venv/bin:$PATH PYTHON=/opt/venv/bin/python
