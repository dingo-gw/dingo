FROM ubuntu:24.04

# Install necessary packages
RUN apt-get update && apt-get install -y \
    git \
    bash \
    jq \
    python3 \
    python3-venv \
    mailutils 

# Copy the script into the container
COPY dingo-ci /usr/local/bin/
RUN chmod +x /usr/local/bin/dingo-ci

# Set the working directory to /usr/local/bin for simplicity
WORKDIR /usr/local/bin

# Set dingo-ci as the entry point
ENTRYPOINT ["dingo-ci"]


