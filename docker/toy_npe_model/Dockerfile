FROM dingo:latest

COPY run_toy_npe_model.bash /opt/dingo/examples/toy_npe_model/

RUN chmod +x /opt/dingo/examples/toy_npe_model/run_toy_npe_model.bash

ENTRYPOINT ["/bin/bash", "-c", ". /opt/dingo/venv/bin/activate && cd /opt/dingo/examples/toy_npe_model && ./run_toy_npe_model.bash"]
