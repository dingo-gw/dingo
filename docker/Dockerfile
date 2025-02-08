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
COPY run_toy_npe_model.bash /usr/local/bin/run_toy_npe_model.bash
RUN chmod +x /usr/local/bin/run_toy_npe_model.bash

# Command to run the script
CMD ["/usr/local/bin/run_toy_npe_model.bash"]
