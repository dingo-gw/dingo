import os

os.environ['OMP_NUM_THREADS'] = str(1)
os.environ['MKL_NUM_THREADS'] = str(1)

import yaml
import argparse

from dingo.api import build_dataset, build_train_and_test_loaders, \
    build_posterior_model, resubmit_condor_job


def parse_args():
    parser = argparse.ArgumentParser(description='Train Dingo.')
    parser.add_argument(
        "--settings_file",
        type=str,
        required=True,
        help="YAML file containing training settings",
    )
    parser.add_argument('--train_dir', required=True,
                        help='Directory for Dingo training output and logging.')
    return parser.parse_args()


def main():

    args = parse_args()
    with open(args.settings_file, 'r') as fp:
        train_settings = yaml.safe_load(fp)

    wfd = build_dataset(train_settings)
    train_loader, test_loader = build_train_and_test_loaders(train_settings, wfd)
    pm = build_posterior_model(args.train_dir, train_settings, wfd[0])

    # train model
    pm.train(
        train_loader,
        test_loader,
        train_dir=args.train_dir,
        runtime_limits_kwargs=train_settings['train_settings']['runtime_limits'],
        checkpoint_epochs=train_settings['train_settings']['checkpoint_epochs'],
    )

    # resubmit condor job
    resubmit_condor_job(
        train_dir=args.train_dir,
        train_settings=train_settings,
        epoch=pm.epoch
    )
