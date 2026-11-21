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

- create a dedicated run directory under the base directory (named after the checkout ref or a timestamp)
- create a virtual environment inside that run directory
- perform a new clone of dingo
- optional: perform a checkout (if the `--checkout` argument is used)
- pip install dingo
- run the toy npe example
- optional: send an email
- delete the virtual environment

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

## Set a continuous integration server

The CI system consists of two independent services:

- **dingo-ci** -- checks for new commits and tags on the main branch every 5 minutes and runs CI when changes are detected. Also runs every Sunday night.
- **dingo-ci-email** -- checks for email-triggered CI requests every 5 minutes and runs CI for the requested commit.

Each service has its own systemd unit and timer, making them independently observable and restartable.

To set it up:

- build the dingo:toy_npe_model docker image (see above)
- copy scripts to `/usr/local/bin`:
  - `dingo-ci-trigger` (commit/tag checking)
  - `dingo-ci-email-trigger`, `email_helpers.py`, `fetch_matching_email.py`,
    `mark_email_seen.py`, `send_ack_email.py` (email-triggered CI)
- copy service and timer files to `/etc/systemd/system`:
  - `dingo-ci.service`, `dingo-ci.timer`
  - `dingo-ci-email.service`, `dingo-ci-email.timer`
- reload systemctl:

```bash
systemctl daemon-reload
```

- enable both services (so that they start at boot):

```bash
systemctl enable dingo-ci.timer
systemctl enable dingo-ci-email.timer
```

- start both timers:

```bash
systemctl start dingo-ci.timer
systemctl start dingo-ci-email.timer
```

- check the status of each service:

```bash
systemctl status dingo-ci.timer
systemctl status dingo-ci-email.timer
```

- monitor for failures:

```bash
# check for failed runs
systemctl --failed | grep dingo-ci
# view logs
journalctl -u dingo-ci.service
journalctl -u dingo-ci-email.service
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

### Debug mode

To test the email trigger without actually running the CI job, add `--debug` to the subject:

```
commit a1b2c3d --debug
```

In debug mode, the system performs all steps (ACK email, folder deletion, marking as read) but instead of running Docker, it creates a job folder with `log.txt` and `error.txt` containing the Docker command that would have been executed, plus a copy of the email trigger logs.

### Behavior

- The `dingo-ci-email` service checks for trigger emails every 5 minutes, independently from the commit/tag service.
- All unread emails are checked in chronological order (oldest first) and marked as read.
- Only one email-triggered job runs per cycle. Remaining trigger emails are processed in subsequent cycles.
- On finding a matching email:
  1. An acknowledgment email is sent to the recipients list.
  2. Any existing job folder for that commit is deleted.
  3. The CI job runs via Docker (same as commit/tag triggered runs).
- Emails with non-matching subjects are marked as read and skipped.
- Email failures are visible to systemd and can be monitored via `journalctl -u dingo-ci-email.service`.

### Logging

The email trigger service maintains a log file at `<JOBS_PATH>/logs.txt` with timestamped entries for:
- Each email checked (with subject and match status)
- Emails marked as read
- ACK email failures
- Jobs started (or debug mode activations)
- Fetch errors
