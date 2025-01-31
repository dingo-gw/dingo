FROM dingo:latest

COPY run_npe_model.bash /opt/dingo/examples/npe_model/

RUN chmod +x /opt/dingo/examples/npe_model/run_npe_model.bash

ENTRYPOINT ["/bin/bash", "-c", ". /opt/dingo/venv/bin/activate && cd /opt/dingo/examples/npe_model && ./run_npe_model.bash"]
