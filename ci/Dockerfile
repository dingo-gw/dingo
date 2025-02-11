FROM ubuntu:24.04

RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y \
        python3 \
        python3-venv \
        python3-dev \
        python3.9 \
        python3.9-venv \
        python3.9-dev \
        git \
        bash \
        jq \
        mailutils \
        mutt \
        build-essential

# Copy the script into the container
COPY dingo-ci /usr/local/bin/
RUN chmod +x /usr/local/bin/dingo-ci

# Set the working directory to /usr/local/bin for simplicity
WORKDIR /usr/local/bin

# Set dingo-ci as the entry point
ENTRYPOINT ["dingo-ci"]
