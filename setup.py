from setuptools import setup

with open('README.md', 'r') as fh:
    long_description = fh.read()
VERSION = '0.0.1'

setup(name='dingo',
      description='A library for deep bayesian inference with GNPE.',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/dingo-gw/dingo-devel',
      license="MIT",
      version=VERSION,
      packages=['dingo', 'dingo.core', 'dingo.core.nn', 'dingo.gw'],
      package_dir={'dingo': 'dingo'},
      python_requires='>=3.6',
      install_requires=[
          'numpy',
          'torch',
          'nflows',
          'scipy'
      ],
      extras_require={
        "dev": [
            "pytest",
            "pylint",
        ],
      },
      classifiers=[
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent"]
      )
