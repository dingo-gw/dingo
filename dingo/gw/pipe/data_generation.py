import sys

from bilby_pipe.data_generation import create_generation_parser, DataGenerationInput
from bilby_pipe.main import parse_args
from bilby_pipe.utils import log_version_information, logger


def save_data(data):
    """Save as DingoDataset HDF5."""
    pass


def main():
    """Data generation main logic"""
    args, unknown_args = parse_args(sys.argv[1:], create_generation_parser())
    log_version_information()
    data = DataGenerationInput(args, unknown_args)
    # data.save_data_dump()
    save_data(data)
    logger.info("Completed data generation")
