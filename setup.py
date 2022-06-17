from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()
VERSION = "0.0.1"

setup(
    name="dingo",
    description="A library for deep bayesian inference with GNPE.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dingo-gw/dingo-devel",
    license="MIT",
    version=VERSION,
    packages=[
        "dingo",
        "dingo.core",
        "dingo.core.nn",
        "dingo.gw",
        "dingo.api",
    ],
    package_dir={"dingo": "dingo"},
    python_requires=">=3.6",
    install_requires=[
        "numpy",
        "torch",
        "torchvision",
        "nflows",
        "scipy",
        "pyyaml",
        "h5py",
        "bilby",
        "astropy",
        "lalsuite", # use >=7.3 if you run into errors with IMRPhenomXPHM generation
        "sklearn",
        "pesummary",
        "pycondor",
        "gwpy",
        "pycbc",
        "pandas",
        "threadpoolctl",
        "chainconsumer",
        "wandb",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pylint",
            "black",
        ],
    },
    entry_points={
        "console_scripts": [
            "dingo_generate_dataset=dingo.gw.dataset.generate_dataset:main",
            "dingo_generate_dataset_dag=dingo.gw.dataset.generate_dataset_dag:main",
            "dingo_merge_datasets=dingo.gw.dataset.utils:merge_datasets_cli",
            "dingo_build_svd=dingo.gw.dataset.utils:build_svd_cli",
            "dingo_generate_ASD_dataset=dingo.gw.ASD_dataset.generate_dataset:generate_dataset",
            "dingo_train=dingo.gw.training:train_local",
            "dingo_train_condor=dingo.gw.training.train_pipeline_condor:train_condor",
            "dingo_append_training_stage=dingo.gw.training:append_stage",
            "dingo_analyze_event=dingo.gw.inference:analyze_event",
            "dingo_ls=dingo.gw.ls_cli:ls",
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
