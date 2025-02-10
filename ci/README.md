# About

This folder provide the bash script `dingo-ci` which automates execution of the dingo TOY-NPE-MODEL example:

- [Toy NPE Example](https://dingo-gw.readthedocs.io/en/latest/example_toy_npe_model.html)

It also provide to setup a related continuous integration script which will:

- check continuously if a new commit has been added to the main branch
- run dingo-ci in docker if there has been a new commit
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

## Running dingo-ci in docker

This suppose docker is already installed. 
See [docker website](https://docs.docker.com/engine/install/)

Build the docker image

```bash
docker build -t dingo:latest .
```

For running the script, for example:

```bash
docker run --shm-size=16g -v /data/dingo:/data/dingo dingo:toy_npe_model --base-dir /data/dingo --email /data/dingo/email.json
``` 

Note:

- If you would like to check the files created/used by the script, share the base directory (here `/data/dingo`)
- The `email.json` file must be accessible to the docker container (here `/data/dingo` is shared)

## Set a continous integration server

- build the docker image as above
- copy the dingo-ci.service file to /etc/systemd/system
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
systemctl status docker-ci
```

- check the status of the service

```
systemctl status docker-ci
```
