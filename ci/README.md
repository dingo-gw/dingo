# About

This folder provide the bash script `dingo-ci` which automates execution of the dingo TOY-NPE-MODEL example:

- [Toy NPE Example](https://dingo-gw.readthedocs.io/en/latest/example_toy_npe_model.html)

It also provides utilities to setup a related continuous integration script which will:

- check continuously if a new commit to the main branch or a new tag has been added
- run `dingo-ci` in docker if there has been a new commit or a new tag
- send a report email (success or failure in running toy-npe-model)


# How-to

## Running dingo-ci

In a terminal:

```bash
# running dingo-ci in /tmp/dingo
./dingo-ci

# running dingo-ci in a folder of your choice
./dingo-ci --base-dir /data/dingo # or smthg else

# running dingo-ci and sending an email report
./dingo-ci --base-dir /data/dingo --email /path/to/email.json

# running dingo-ci on the development branch
./dingo-ci --checkout development
```

email.json should look like:

```json
{
    "root": "user@domain.eu",
    "mailhub": "domain.eu",
    "port": 465,
    "authUser": "user",
    "authPass": "mypass",
    "recipients": ["myfriend@frienddomain.eu"]
}
```

`dingo-ci` will:

- activate `/tmp/dingo/venv` (create it first if it does not exist)
- perform a new clone of dingo
- optional: perform a checkout (if the `--checkout` argument is used)
- pip install dingo
- run the toy npe example
- optional: send an email

## Using another python/torch

By default, the default python3 and the latest torch will be used. To use other versions:

Uncommand/edit the lines defining the variables `PYTHON_VERSION`
and `TORCH_INSTALL_COMMAND` at the top of `dingo-ci`.

For example:

```bash
PYTHON_VERSION="python3.9"
TORCH_INSTALL_COMMAND="pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu113 --upgrade"
```


## Running dingo-ci in docker

This suppose docker is already installed. 
See [docker website](https://docs.docker.com/engine/install/)

Build the docker image

```bash
docker build -t dingo:toy_npe_model .
```

For running the script, for example:

```bash
docker run --shm-size=16g -v /data/dingo:/data/dingo dingo:toy_npe_model \
            --base-dir /data/dingo --email /data/dingo/email.json
``` 

Notes:

- If you would like to check the files created/used by the script, mount the base directory using '-v' argument (here `/data/dingo`)
- The `email.json` file must be accessible to the docker container (here `/data/dingo` is mounted)

To run with GPU support:

```bash
docker run --rm --shm-size=16g --runtime=nvidia --gpus all -v /data/dingo:/data/dingo dingo:toy_npe_model --base-dir /data/dingo 
```

This requires the [nvidia container toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit) to be installed on the host machine.

## Set a continous integration server

A continuous integration server will use systemctl services to call the `dingo-ci-trigger` bash script every five minutes. This bash script checks if there are new commits (on the main branch) or new tags added to the dingo repository. If so, it will call `dingo-ci` in docker. 

To set it up:

- build the dingo:toy_npe_model docker image (see above)
- copy the `dingo-ci-trigger` script to `/usr/local/bin`.
- copy the `dingo-ci.service` file to `/etc/systemd/system`
- copy the `dingo-ci.timer` file to `/etc/systemd/system`
- reload systemctl:

```bash
systemctl daemon-reload
```

- activate the dingo-ci service (so that it starts at boot):

```bash
systemctl enable docker-ci
```

- start the service

```
systemctl start docker-ci
```

- check the status of the service

```
systemctl status docker-ci
```
