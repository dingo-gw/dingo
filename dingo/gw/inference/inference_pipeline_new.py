import torch

from dingo.core.models import PosteriorModel
from dingo.gw.inference import GWSampler, GWSamplerGNPE, get_event_data
from dingo.gw.inference.gw_samplers import GWSamplerUnconditional


def inference():

    args = parse_args()

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    #
    # 1. Load model(s)
    #

    model = PosteriorModel(args.model, device=device, load_training_info=False)

    if args.model_init is not None:
        gnpe = True
        init_model = PosteriorModel(
            args.model_init, device=device, load_training_info=False
        )
        init_sampler = GWSampler(model=init_model)
        sampler = GWSamplerGNPE(
            model=model,
            init_sampler=init_sampler,
            num_iterations=args.num_gnpe_iterations,
        )
    else:
        gnpe = False
        sampler = GWSampler(model=model)

    #
    # 2. Get strain data.
    #

    print(f"Analyzing event at {time_event}.")
    event_data, event_metadata, label = get_event_data(time_event, args, model, ref)
    sampler.context = event_data
    sampler.event_metadata = event_metadata

    #
    # 3. (Optional) Recover density.
    #   For importance sampling, we require access to the probability density. However,
    #   this is not immediately available if using GNPE. To fix this, we train an
    #   unconditional flow.
    #

    if args.get_log_prob and gnpe:

        # 3.(a) Generate samples

        unconditional_flow_settings = get_unconditional_flow_settings()

        # TODO: Set number of samples based on unconditional_flow_settings.
        sampler.run_sampler(unconditional_flow_settings.num_samples, args.batch_size)
        result = sampler.to_samples_dataset()

        # (Optional) Save low-latency samples.
        if args.save_low_latency:
            result.to_file(label + "_low_latency.hdf5")

        # Free up memory.
        sampler.init_sampler = None
        del init_sampler, init_model

        # 3.(b) Train unconditional flow

        # There are two ways to do this. Either train a 2D or 3D flow for the GNPE
        # proxies, or train a flow for the full parameter space. For the latter,
        # we have to make sure we have recovered the synthetic phase if we wish to do
        # importance sampling.
        unconditional_proxy_flow = result.train_unconditional_flow(proxy_parameters)

        # 3.(c) Define new sampler
        unconditional_proxy_sampler = UnconditionalSampler(
            model=unconditional_proxy_flow)
        # TODO: Make sure this works.
        sampler.gnpe_proxy_sampler = unconditional_proxy_sampler
        # sampler.num_iterations = 1

    #
    # 4. Generate samples
    #

    #
    # 5. (Optional) Recover phase (CPU)
    #

    #
    # 6. (Optional) Importance sample (CPU)
    #
