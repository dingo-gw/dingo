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
      packages=['dingo', 'dingo.core', 'dingo.core.nn', 'dingo.gw',
                'dingo.api',],
      package_dir={'dingo': 'dingo'},
      python_requires='>=3.6',
      install_requires=[
          'numpy',
          'torch',
          'torchvision',
          'nflows',
          'scipy',
          'pyyaml',
          'h5py',
          'bilby',
          'astropy',
          'lalsuite',
          'sklearn',
          'pycondor'
      ],
      extras_require={
          "dev": [
              "pytest",
              "pylint",
          ],
      },
      entry_points={'console_scripts':
          ['generate_parameters=tutorials.02_gwpe.generate_parameters:main',
           'generate_waveforms=tutorials.02_gwpe.generate_waveforms:main',
           'build_SVD_basis=tutorials.02_gwpe.build_SVD_basis:main',
           'collect_waveform_dataset=tutorials.02_gwpe.collect_waveform_dataset:main',
           'create_waveform_generation_bash_script=tutorials.02_gwpe.create_waveform_generation_bash_script:main',
           'create_waveform_generation_dag=tutorials.02_gwpe.create_waveform_generation_dag:main']
      },
      classifiers=[
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent"]
      )
