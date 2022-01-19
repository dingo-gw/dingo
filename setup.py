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
          'pycondor',
          'gwpy',
          'pycbc',
          'pandas',
      ],
      extras_require={
          "dev": [
              "pytest",
              "pylint",
          ],
      },
      entry_points={'console_scripts':
          ['generate_parameters=dingo.gw.dataset_generation.generate_parameters:main',
           'generate_waveforms=dingo.gw.dataset_generation.generate_waveforms:main',
           'build_SVD_basis=dingo.gw.dataset_generation.build_SVD_basis:main',
           'collect_waveform_dataset=dingo.gw.dataset_generation.collect_waveform_dataset:main',
           'create_waveform_generation_bash_script=dingo.gw.dataset_generation.create_waveform_generation_bash_script:main',
           'create_waveform_generation_dag=dingo.gw.dataset_generation.create_waveform_generation_dag:main',
           'generate_dataset=dingo.gw.dataset_generation.generate_dataset:main',
           'dingo_generate_dataset_dag=dingo.gw.dataset_generation.generate_dataset_dag:main']
      },
      classifiers=[
          "Programming Language :: Python :: 3.6",
          "Programming Language :: Python :: 3.7",
          "License :: OSI Approved :: MIT License",
          "Operating System :: OS Independent"]
      )
