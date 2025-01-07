#
#  Adapted from bilby_pipe.
#

import copy

from bilby_pipe.job_creation.dag import Dag
from bilby_pipe.utils import BilbyPipeError, logger

from dingo.pipe.utils import _strip_unwanted_submission_keys
from dingo.pipe.nodes.generation_node import GenerationNode
from .nodes.importance_sampling_node import ImportanceSamplingNode
from .nodes.merge_node import MergeNode
from .nodes.pe_summary_node import PESummaryNode
from .nodes.plot_node import PlotNode
from .nodes.sampling_node import SamplingNode

logger.name = "dingo_pipe"


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


def get_parallel_list(inputs):
    if inputs.n_parallel == 1:
        return [""]
    else:
        return [f"part{idx}" for idx in range(inputs.n_parallel)]


def generate_dag(inputs, model_args):
    inputs = copy.deepcopy(inputs)
    dag = Dag(inputs)
    trigger_times = get_trigger_time_list(inputs)

    if inputs.simple_submission:
        _strip_unwanted_submission_keys(dag.pycondor_dag)
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

    if inputs.importance_sample:
        #
        # 3. Generate new data for importance sampling **if different settings requested**.
        #
        # If injecting into simulated noise, be sure to use consistent noise realization.

        if len(inputs.importance_sampling_updates) > 0:
            # Iterate over all generation nodes and store them in a list
            importance_sampling_generation_node_list = []
            for idx, trigger_time in enumerate(trigger_times):
                kwargs = dict(trigger_time=trigger_time, idx=idx, dag=dag)
                if idx > 0:
                    # Make all generation nodes depend on the 0th generation node
                    # Ensures any cached files (e.g. the distance-marginalization
                    # lookup table) are only built once.
                    kwargs["parent"] = generation_node_list[0]
                generation_node = GenerationNode(inputs, importance_sampling=True, **kwargs)
                importance_sampling_generation_node_list.append(generation_node)
        else:
            importance_sampling_generation_node_list = generation_node_list

        #
        # 4. Importance sample
        #
        # If the phase is not present and phase marginalization is not being used, also
        # sample the phase synthetically. This adds between 1x and 50x to the cost of
        # importance sampling, depending on the waveform model. Indeed, IMRPhenomXPHM
        # waveform modes are much more expensive to generate than polarizations.

        merged_importance_sampling_node_list = []
        parallel_list = get_parallel_list(inputs)
        all_parallel_node_list = []
        for sampling_node, generation_node in zip(
            sampling_node_list, importance_sampling_generation_node_list
        ):
            parallel_node_list = []
            for parallel_idx in parallel_list:
                importance_sampling_node = ImportanceSamplingNode(
                    inputs,
                    sampling_node=sampling_node,
                    generation_node=generation_node,
                    parallel_idx=parallel_idx,
                    dag=dag,
                )
                parallel_node_list.append(importance_sampling_node)
                all_parallel_node_list.append(importance_sampling_node)

            if len(parallel_node_list) == 1:
                merged_importance_sampling_node_list.append(importance_sampling_node)
            else:
                # 4.(b) Recombine jobs into single Result.
                #       (Automatically calculates evidence.)
                merge_node = MergeNode(
                    inputs=inputs,
                    parallel_node_list=parallel_node_list,
                    dag=dag,
                )
                merged_importance_sampling_node_list.append(merge_node)

    else:
        merged_importance_sampling_node_list = sampling_node_list

    #
    # 5. Plotting
    #

    plot_nodes_list = []
    for merged_node in merged_importance_sampling_node_list:
        if inputs.plot_node_needed:
            plot_nodes_list.append(PlotNode(inputs, merged_node, dag=dag))

    #
    # 6. PESummary
    #

    if inputs.create_summary:
        # Add the waveform approximant to inputs, so that it can be fed to PESummary.
        inputs.waveform_approximant = model_args["waveform_approximant"]
        PESummaryNode(inputs, merged_importance_sampling_node_list, generation_node_list, dag=dag)

    dag.build()
    # create_overview(
    #     inputs,
    #     generation_node_list,
    #     all_parallel_node_list,
    #     merged_node_list,
    #     plot_nodes_list,
    # )
