# CI image for dnf-based cells (alma 9, manylinux_2_28 = alma 8); build args
# come from ci/image.py.  PY_ENSURE asks manylinux for one of its /opt/python builds.
ARG BASE
FROM ${BASE}
LABEL org.opencontainers.image.source=https://github.com/lijinpei-amd/intj
ARG PY_PKGS
ARG PY_ENSURE
ARG PY_BIN
ARG TORCH_INDEX
ARG TRITON
RUN dnf install -y gcc gcc-c++ git util-linux ${PY_PKGS} && dnf clean all
RUN if [ -n "$PY_ENSURE" ]; then manylinux-interpreters ensure "$PY_ENSURE"; fi
COPY install-venv.sh /tmp/install-venv.sh
RUN sh /tmp/install-venv.sh
ENV PATH=/opt/venv/bin:$PATH PYTHON=/opt/venv/bin/python
# where `TORCH=<version> ninja -C ci install` gets another torch from
ENV INTJ_TORCH_INDEX=${TORCH_INDEX}
