# About

This folder contains files for running Dingo examples in a docker container.
Are currently supported:

- [Toy NPE Example](https://dingo-gw.readthedocs.io/en/latest/example_toy_npe_model.html) in a docker container.
- [NPE Model](https://dingo-gw.readthedocs.io/en/latest/example_npe_model.html)

# How to

## install docker

See [docker website](https://docs.docker.com/engine/install/)

## build the 'main' docker image

```bash
docker build -t dingo:latest .
```

## run the examples

### visit the desired subfolder

e.g.

```bash
cd toy_npe_model
```

### create the corresponding docker image

```bash
docker build -t dingo_toy_npe_model:latest .
```

### run the image


```
# 16g is the amount of shared memory the container will
# be allowed to use. You can change this value.
# Note that failing to use this option may result in a torch error
# advising "Please try to raise your shared memory limit"

# -v /tmp:/tmp will bind the container /tmp folder to the local /tmp
# folder, where all the output files will be created.

docker run --shm-size=16g -v /tmp:/tmp dingo_toy_npe_model:latest
```

