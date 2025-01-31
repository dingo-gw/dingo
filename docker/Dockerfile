FROM ubuntu:24.04

RUN apt-get update && apt-get install -y \
    git \
    python3 \
    python3-pip \
    python3-venv

RUN git clone https://github.com/dingo-gw/dingo.git /opt/dingo && \
    cd /opt/dingo && \
    python3 -m venv ./venv && \
    . ./venv/bin/activate && \
    pip install .

COPY dingo.bash /opt/dingo/docker/
