#
#  Adapted from bilby_pipe.
#

import copy

from bilby_pipe.job_creation.bilby_pipe_dag_creator import get_detectors_list
from bilby_pipe.job_creation.dag import Dag
from bilby_pipe.job_creation.overview import create_overview
from bilby_pipe.utils import BilbyPipeError, logger

from .generation_node import GenerationNode
from .nodes.sampling_node import SamplingNode


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

    #
    # 1. Generate data for inference.
    #

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

    #
    # 2. Sample the posterior using Dingo.
    #
    # Reconstruct the posterior density if necessary. This requires training a new
    # network and sampling from it.

    sampling_node_list = []
    for generation_node in generation_node_list:
        sampling_node = SamplingNode(
            inputs,
            generation_node=generation_node,
            dag=dag,
        )
        sampling_node_list.append(sampling_node)

    #
    # 3. Generate new data for importance sampling **if different settings requested**.
    #
    # If injecting into simulated noise, be sure to use consistent noise realization.

    #
    # 4. Importance sample
    #

    # 4.(a) (If necessary) Split the proposal samples into sub-Results for
    # parallelization across jobs.

    # 4.(b) Calculate importance weights.
    #
    # If the phase is not present and phase marginalization is not being used,
    # sample the phase synthetically. This adds between 1x and 50x to the cost of
    # importance sampling, depending on the waveform model. Indeed, IMRPhenomXPHM
    # waveform modes are much more expensive to generate than polarizations.

    # 4.(c) (If necessary) Recombine jobs into single Result.

    # 4.(d) Calculate evidence.  POSSIBLY COMBINE WITH 4B IF ONLY ONE JOB.

    #
    # 5. PESummary
    #


    dag.build()
    # create_overview(
    #     inputs,
    #     generation_node_list,
    #     all_parallel_node_list,
    #     merged_node_list,
    #     plot_nodes_list,
    # )
