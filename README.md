
> **Notice:** This code is under development, and the authors plan to use it for several short-author papers before making it widely available. Feel free to peruse, use, or experiment with this code, but please do not distribute beyond the LSC. If you wish to publish work based on this code or the unpublished ideas therein, please contact [Stephen Green](mailto:stephen.green@aei.mpg.de) and [Maximilian Dax](mailto:maximilian.dax@tuebingen.mpg.de) beforehand. Comments are welcome!

# Dingo: Deep inference for gravitational-wave observations

This code is based on the inference framework described in https://arxiv.org/abs/2106.12594.

# Developing dingo

To install dingo, along with the tools for development and testing, do the following:

Create and activate a virtual environment. By convention, the environment is called `venv` and is located in the `dingo-devel` directory. Some unit tests depend on this convention.

```bash
$ python3 -m venv venv
$ source venv/bin/activate
```

In this virtual environment, install dingo.

```bash
$ pip install wheel
$ python setup.py bdist_wheel
$ pip install -e ."[dev]"
```
