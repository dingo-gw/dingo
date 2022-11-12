#
#  Adapted from bilby_pipe.
#

import copy

from bilby_pipe.job_creation.dag import Dag
from bilby_pipe.utils import BilbyPipeError, logger

from .generation_node import GenerationNode


def get_trigger_time_list(inputs):
    """Returns a list of GPS trigger times for each data segment"""
    # if (inputs.gaussian_noise or inputs.zero_noise) and inputs.trigger_time is None:
    #     trigger_times = [0] * inputs.n_simulation
    # elif (inputs.gaussian_noise or inputs.zero_noise) and isinstance(
    #     inputs.trigger_time, float
    # ):
    #     trigger_times = [inputs.trigger_time] * inputs.n_simulation
    if inputs.trigger_time is not None:
        trigger_times = [inputs.trigger_time]
    elif getattr(inputs, "gpstimes", None) is not None:
        start_times = inputs.gpstimes
        trigger_times = start_times + inputs.duration - inputs.post_trigger_duration
    else:
        raise BilbyPipeError("Unable to determine input trigger times from ini file")
    logger.info(f"Setting segment trigger-times {trigger_times}")
    return trigger_times


def generate_dag(inputs):
    inputs = copy.deepcopy(inputs)
    dag = Dag(inputs)
    trigger_times = get_trigger_time_list(inputs)

    # Iterate over all generation nodes and store them in a list
    generation_node_list = []
    for idx, trigger_time in enumerate(trigger_times):
        kwargs = dict(trigger_time=trigger_time, idx=idx, dag=dag)
        if idx > 0:
            # Make all generation nodes depend on the 0th generation node
            # Ensures any cached files (e.g. the distance-marginalization
            # lookup table) are only built once.
            kwargs["parent"] = generation_node_list[0]
        generation_node = GenerationNode(inputs, **kwargs)
        generation_node_list.append(generation_node)

    # breakpoint()

    return
