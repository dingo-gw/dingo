# dingo-devel
Development code for Dingo: Deep inference for gravitational-wave observations

# Developing dingo

To install dingo, along with the tools for development and testing, do the following:

Create and activate a virtual environment.

```bash
$ python3 -m venv dingo-devenv
$ source dingo-devenv/bin/activate
```

In this virtual environment, install dingo.

```bash
$ pip install wheel
$ python3 setup.py bdist_wheel
$ pip install -e ."[dev]"
```
