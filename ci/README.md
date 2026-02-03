# About

This folder provide the bash script `dingo-ci` which automates execution of the dingo TOY-NPE-MODEL example:

- [Toy NPE Example](https://dingo-gw.readthedocs.io/en/latest/example_toy_npe_model.html)

It also provides utilities to setup a related continuous integration script which will:

- check continuously if a new commit to the main branch or a new tag has been added
- run `dingo-ci` in docker if there has been a new commit or a new tag
- send a report email (success or failure in running toy-npe-model)


# How-to

## Running dingo-ci

`dingo-ci` is a standalone bash script. Dingo does not need to be installed for it to run.

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
    "recipients": ["myfriend@frienddomain.eu"],
    "imap": {
        "server": "domain.eu",
        "port": 993
    }
}
```

The `imap` section is optional and only required for [email-triggered CI runs](#email-triggered-ci-runs).

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

A continuous integration server will use systemctl services to call the `dingo-ci-trigger` bash script every five minutes. This bash script checks if there are new commits (on the main branch) or new tags added to the dingo repository. If so, it will call `dingo-ci` in docker. It calls also `dingo-ci` in docker every sunday night.

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
systemctl enable dingo-ci
```

- start the service

```
systemctl start docker-ci
```

- check the status of the service

```
systemctl status docker-ci
```

## Email-triggered CI runs

In addition to automatic commit/tag detection, the CI system can be triggered
by sending an email to the CI account.

### Setup

Add IMAP settings to your `email.json` (see above). The CI system reuses
`authUser` and `authPass` for IMAP authentication. Port 993 is IMAP over SSL/TLS.

The host machine must have `python3` available (standard library only, no extra packages).

### Triggering a run

Send an email to the CI account (`root` address in the config) with the subject:

```
commit <commit_hash>
```

where `<commit_hash>` is a 7-40 character hex SHA (short or full).
For example: `commit a1b2c3d` or `commit a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4e5f6a1b2`.

### Behavior

- The CI system checks for trigger emails every 5 minutes (each time `dingo-ci-trigger` runs).
- Unread emails are checked in chronological order (oldest first).
- Only one email-triggered job runs per cycle. Remaining emails are processed in subsequent cycles.
- On finding a matching email:
  1. An acknowledgment email is sent to the CI account itself.
  2. Any existing job folder for that commit is deleted.
  3. The CI job runs via Docker (same as commit/tag triggered runs).
  4. The trigger email is marked as read.
- Emails with non-matching subjects are left unread and ignored.
- After email checking, the normal commit/tag detection continues as usual.
- Email checking failures are non-fatal: the system falls back to normal commit/tag detection.
