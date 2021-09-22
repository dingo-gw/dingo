"""
This is an implementation of parameter inference for a toy example. We build
a toy model based on the harmonic oscillator with similar properties as the
GW parameter estimation problem. We then use (G)NPE to perform parameter
inference.

As forward model we use a harmonic oscillator that is initially at rest,
and then excited with an infinitesimally short pulse. After excitation,
there is a damped oscillation.

The forward model has 2 intrinsic parameters:
    omega0 :    resonance frequency of undamped oscillator
    beta :      damping factor

To build an example similar to the GW use case, the corresponding observation
is a pair of time series in Nd different detectors. The signal arrives in
Detector i at time ti, which is sampled for each detector at train time. The
model has thus Nd extrinsic parameters. In total, the forward model has 2 +
Nd parameters {t, omega0, beta, t0, t1, ...}.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
from toy_example_utils import *
import yaml
from os.path import join
from dingo.core.nn.nsf import create_nsf_with_rb_projection_embedding_net
from dingo.core.models import PosteriorModel
import h5py


if __name__ == '__main__':
    model_builder = create_nsf_with_rb_projection_embedding_net
    transform_builder = get_transformations_composites
    log_dir = '../../logs/toy_example/01/'
    model_path = join(log_dir, 'model_latest.pt')
    dataset_path = join(log_dir, 'dataset.hdf5')
    initialize = True
    if initialize:
        # Read YAML file
        with open(join(log_dir, 'config.yaml'), 'r') as stream:
            config = yaml.safe_load(stream)

        t_axis = config['simulator_settings']['t_axis']
        simulator = HarmonicOscillator(*t_axis)

        # get dataset for extrinsic parameters
        extrinsic_prior = config['simulator_settings']['extrinsic_prior']
        intrinsic_prior = config['simulator_settings']['intrinsic_prior']
        num_samples = config['simulator_settings']['size_dataset']
        prior = extrinsic_prior + intrinsic_prior
        parameters = sample_from_uniform_prior(
            extrinsic_prior, num_samples=num_samples)
        simulations = simulator.simulate(parameters)

        # build/load the dataset
        with h5py.File(dataset_path, 'w') as f:
            f.create_dataset('parameters', data=parameters)
            f.create_dataset('simulations', data=simulations)
        dataset_kwargs = {'dataset_filename': dataset_path}

        # build the transformation kwargs
        projection_kwargs = {
            'num_detectors': len(intrinsic_prior),
            'num_channels': 3,
            't_axis_delta': simulator.delta_t,
            'ti_priors_list': intrinsic_prior,
        }
        noise_module_kwargs = {
            'num_bins': simulator.N_bins,
            'std': 0.01,
        }
        normalization_kwargs = {
            'means': get_means_and_stds_from_uniform_prior_ranges(prior)[0],
            'stds': get_means_and_stds_from_uniform_prior_ranges(prior)[1],
        }
        transformation_kwargs = {
            'projection_kwargs': projection_kwargs,
            'noise_module_kwargs': noise_module_kwargs,
            'normalization_kwargs': normalization_kwargs,
        }
        train_transform, *_ = transform_builder(**transformation_kwargs)

        # build the model
        nsf_kwargs = config['model_config']['nsf_kwargs']
        nsf_kwargs['input_dim'] = len(prior)
        embedding_net_kwargs = config['model_config']['embedding_net_kwargs']
        embedding_net_kwargs['input_dims'] = \
            (len(extrinsic_prior), 3, simulator.N_bins)
        embedding_net_kwargs['V_rb_list'] = None
        model_kwargs = {
            'nsf_kwargs': nsf_kwargs,
            'embedding_net_kwargs': embedding_net_kwargs,
        }
        pm = PosteriorModel(
            model_builder=model_builder,
            model_kwargs=model_kwargs,
            optimizer_kwargs=config['model_config']['optimizer_kwargs'],
            scheduler_kwargs=config['model_config']['scheduler_kwargs'],
            init_for_training=True,
            transform_kwargs=transformation_kwargs,
            dataset_kwargs=dataset_kwargs,
        )

        pm.save_model(model_filename=model_path,
                      save_training_info=True)


    pm_loaded = PosteriorModel(
        model_builder=model_builder,
        model_filename=model_path,
        init_for_training=True,
    )

    pm_loaded.initialize_dataloader(SimulationsDataset, transform_builder)


    # set up the simulator


    # set up forward model
    t_axis = config['simulator_settings']['t_axis']
    simulator = HarmonicOscillator(*t_axis)

    # get dataset for extrinsic parameters
    extrinsic_prior = [[3.0, 10.0], [0.2, 0.5]]
    intrinsic_prior = [[0, 3], [2, 5]]
    prior = extrinsic_prior + intrinsic_prior
    parameters = sample_from_uniform_prior(extrinsic_prior,
                                           num_samples=5_000)
    simulations = simulator.simulate(parameters)
    dataset = SimulationsDataset(parameters=parameters, simulations=simulations)

    ###################
    # transformations #
    ###################

    projection_kwargs = {
        'num_detectors': 2,
        'num_channels': 3,
        't_axis_delta': simulator.delta_t,
        'ti_priors_list': intrinsic_prior,
    }
    noise_module_kwargs = {
        'num_bins': simulator.N_bins,
        'std': 0.01,
    }
    normalization_kwargs = {
        'means': get_means_and_stds_from_uniform_prior_ranges(prior)[0],
        'stds': get_means_and_stds_from_uniform_prior_ranges(prior)[1],
    }

    transform_projection = ProjectOntoDetectors(**projection_kwargs)

    noise_module = NoiseModule(**noise_module_kwargs)
    transform_whiten_signal_and_add_noise_summary = \
        WhitenSignalAndGetNoiseSummary(noise_module)
    transform_add_white_noise = AddWhiteNoise(noise_module)

    transform_normalize_parameters = NormalizeParameters(
        **normalization_kwargs, inverse=False)
    transform_denormalize_parameters = NormalizeParameters(
        **normalization_kwargs, inverse=True)

    # get data from dataset
    data = dataset[0]
    print(data['parameters'])
    plt.title('Raw data')
    plt.plot(simulator.t_axis, data['simulations'])
    plt.show()

    # sample extrinsic parameters, project onto the detectors
    data = transform_projection(data)
    print(data['parameters'])
    for idx in range(data['simulations'].shape[0]):
        plt.title('Projection onto detector {:}'.format(idx))
        plt.plot(simulator.t_axis, data['simulations'][idx, 0],
                 label='real')
        plt.plot(simulator.t_axis, data['simulations'][idx, 1],
                 label='imag')
        plt.plot(simulator.t_axis, data['simulations'][idx, 2])
        plt.legend()
        plt.show()

    # draw noise distribution, whiten and add context data
    data = transform_whiten_signal_and_add_noise_summary(data)
    print(data['parameters'])
    for idx in range(data['simulations'].shape[0]):
        plt.title('Whitened data in detector i {:}'.format(idx))
        plt.plot(simulator.t_axis, data['simulations'][idx, 0],
                 label='real')
        plt.plot(simulator.t_axis, data['simulations'][idx, 1],
                 label='imag')
        plt.legend()
        plt.show()
        plt.title('Noise summary in detector i {:}'.format(idx))
        plt.plot(simulator.t_axis, data['simulations'][idx, 2])
        plt.show()

    # add white noise to whitened signal
    data = transform_add_white_noise(data)
    print(data['parameters'])
    for idx in range(data['simulations'].shape[0]):
        plt.title('Noisy data in detector i {:}'.format(idx))
        plt.plot(simulator.t_axis, data['simulations'][idx, 0],
                 label='real')
        plt.plot(simulator.t_axis, data['simulations'][idx, 1],
                 label='imag')
        plt.legend()
        plt.show()
        plt.title('Noise summary in detector i {:}'.format(idx))
        plt.plot(simulator.t_axis, data['simulations'][idx, 2])
        plt.show()

    data = transform_normalize_parameters(data)
    print(data['parameters'])
    transform_denormalize_parameters(data)
    print(transform_denormalize_parameters(data)['parameters'])

    transformation_kwargs = {
        'projection_kwargs': projection_kwargs,
        'noise_module_kwargs': noise_module_kwargs,
        'normalization_kwargs': normalization_kwargs,
    }
    transform_full, _, inference_postprocessing, _ = \
        get_transformations_composites(**transformation_kwargs)

    data_transformed = transform_full(dataset[0])

    print(data['parameters'])
    print(data_transformed['parameters'])
    for idx in range(data['simulations'].shape[0]):
        plt.title('Noisy data in detector i {:}'.format(idx))
        plt.plot(simulator.t_axis, data['simulations'][idx, 0],
                 label='real')
        plt.plot(simulator.t_axis, data['simulations'][idx, 1],
                 label='imag')
        plt.plot(simulator.t_axis, data_transformed['simulations'][idx, 0],
                 label='real')
        plt.plot(simulator.t_axis, data_transformed['simulations'][idx, 1],
                 label='imag')
        plt.legend()
        plt.show()
        plt.title('Noise summary in detector i {:}'.format(idx))
        plt.plot(simulator.t_axis, data['simulations'][idx, 2])
        plt.plot(simulator.t_axis, data_transformed['simulations'][idx, 2])
        plt.show()

        pass
