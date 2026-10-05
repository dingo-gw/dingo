FROM ubuntu:24.04

RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-venv \
        python3.11-dev \
        git \
        bash \
        jq \
        mailutils \
        mutt \
        build-essential && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 110 && \
    update-alternatives --set python3 /usr/bin/python3.11 && \
    ln -sf /usr/bin/python3.11 /usr/local/bin/python && \
    rm -rf /var/lib/apt/lists/*

COPY dingo-ci /usr/local/bin/
RUN chmod +x /usr/local/bin/dingo-ci

WORKDIR /usr/local/bin
ENTRYPOINT ["dingo-ci"]
