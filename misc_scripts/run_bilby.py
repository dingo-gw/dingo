#!/usr/bin/env python
"""
Adapted from https://git.ligo.org/lscsoft/bilby/blob/master/examples/gw_examples/data_examples/GW150914.py

Tutorial to demonstrate running parameter estimation on GW150914

This example estimates all 15 parameters of the binary black hole system using
commonly used prior distributions. This will take several hours to run. The
data is obtained using gwpy, see [1] for information on how to access data on
the LIGO Data Grid instead.

[1] https://gwpy.github.io/docs/stable/timeseries/remote-access.html
"""
import argparse
import bilby
from os.path import join
from gwpy.timeseries import TimeSeries
import pycbc.psd
from scipy.signal import tukey
import numpy as np
import yaml
from shutil import copyfile

default_extrinsic_dict = {
    "dec": "bilby.core.prior.Cosine(minimum=-np.pi/2, maximum=np.pi/2)",
    "ra": 'bilby.core.prior.Uniform(minimum=0., maximum=2*np.pi, boundary="periodic")',
    "geocent_time": "bilby.core.prior.Uniform(minimum=-0.1, maximum=0.1)",
    "psi": 'bilby.core.prior.Uniform(minimum=0.0, maximum=np.pi, boundary="periodic")',
    "luminosity_distance": "bilby.core.prior.Uniform(minimum=100.0, maximum=6000.0)",
}

default_intrinsic_dict = {
    "mass_1": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)",
    "mass_2": "bilby.core.prior.Constraint(minimum=10.0, maximum=80.0)",
    "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=1.0)",
    "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=25.0, maximum=100.0)",
    "luminosity_distance": 1000.0,
    "theta_jn": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
    "phase": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
    "a_1": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)",
    "a_2": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)",
    "tilt_1": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
    "tilt_2": "bilby.core.prior.Sine(minimum=0.0, maximum=np.pi)",
    "phi_12": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
    "phi_jl": 'bilby.core.prior.Uniform(minimum=0.0, maximum=2*np.pi, boundary="periodic")',
    "geocent_time": 0.0,
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run bilby.",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        required=True,
        help="Directory for output of plots.",
    )
    parser.add_argument(
        "--settings_file",
        type=str,
        default=None,
        help="Yaml file with bilby settings. If None, use bilby_settings.yaml in "
        "args.outdir.",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":

    args = parse_args()
    logger = bilby.core.utils.logger
    outdir = args.outdir
    bilby.core.utils.check_directory_exists_and_if_not_mkdir(outdir)
    if args.settings_file is not None:
        copyfile(args.settings_file, join(outdir, "bilby_settings.yaml"))
    with open(join(outdir, "bilby_settings.yaml")) as f:
        settings = yaml.safe_load(f)

    duration = settings["duration"]
    end_time = settings["trigger_time"] + settings["post_trigger_duration"]
    start_time = end_time - duration
    print(f"Start time: {start_time}, end time: {end_time}")
    psd_start_time = start_time - settings["psd_duration"]
    psd_end_time = start_time
    print(f"psd start time: {psd_start_time}, psd end time: {psd_end_time}")

    # We now use gwpy to obtain analysis and psd data and create the ifo_list
    ifo_list = bilby.gw.detector.InterferometerList([])
    for det in settings["detectors"]:
        logger.info("Downloading analysis data for ifo {}".format(det))
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        ifo.maximum_frequency = settings["maximum_frequency"]
        ifo.minimum_frequency = settings["minimum_frequency"]
        ifo.strain_data.roll_off = settings["roll_off"]  # Default is 0.2.
        data = TimeSeries.fetch_open_data(det, start_time, end_time, cache=True)
        ifo.strain_data.set_from_gwpy_timeseries(data)

        logger.info("Downloading psd data for ifo {}".format(det))
        psd_data = TimeSeries.fetch_open_data(
            det, psd_start_time, psd_end_time, cache=True
        )

        psd_alpha = 2 * settings["roll_off"] / duration

        sampling_rate = len(psd_data) / settings["psd_duration"]
        psd_data_pycbc = psd_data.to_pycbc()
        w = tukey(int(duration * sampling_rate), psd_alpha)
        psd = pycbc.psd.estimate.welch(
            psd_data_pycbc,
            seg_len=int(duration * sampling_rate),
            seg_stride=int(duration * sampling_rate),
            window=w,
            avg_method="median",
        )
        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(
            frequency_array=np.array(psd.sample_frequencies), psd_array=np.array(psd)
        )

        ifo_list.append(ifo)

    logger.info("Saving data plots to {}".format(outdir))
    ifo_list.plot_data(outdir=outdir, label=settings["event_label"])

    ifo_list.save_data(outdir, settings["event_label"])

    # We now define the prior.
    # The prior is printed to the terminal at run-time.
    # You can overwrite this using the syntax below in the file,
    # or choose a fixed value by just providing a float value as the prior.
    prior = {
        k: {**default_intrinsic_dict, **default_extrinsic_dict}[k]
        if v == "default"
        else v
        for k, v in settings["prior"].items()
    }
    prior = bilby.gw.prior.BBHPriorDict(prior)
    prior["geocent_time"].minimum += settings["trigger_time"]
    prior["geocent_time"].maximum += settings["trigger_time"]

    # In this step we define a `waveform_generator`. This is the object which
    # creates the frequency-domain strain. In this instance, we are using the
    # `lal_binary_black_hole model` source model. We also pass other parameters:
    # the waveform approximant and reference frequency and a parameter conversion
    # which allows us to sample in chirp mass and ratio rather than component mass
    waveform_generator = bilby.gw.WaveformGenerator(
        sampling_frequency=sampling_rate,
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments={
            "waveform_approximant": settings["waveform_model"],
            "reference_frequency": settings["reference_frequency"],
        },
    )

    # In this step, we define the likelihood. Here we use the standard likelihood
    # function, passing it the data and the waveform generator.
    # Note, phase_marginalization is formally invalid with a precessing waveform such as IMRPhenomPv2
    likelihood = bilby.gw.likelihood.GravitationalWaveTransient(
        ifo_list,
        waveform_generator,
        priors=prior,
        time_marginalization=settings.get("time_marginalization", True),
        phase_marginalization=settings.get("phase_marginalization", False),
        distance_marginalization=settings.get("distance_marginalization", True),
    )

    # Finally, we run the sampler. This function takes the likelihood and prior
    # along with some options for how to do the sampling and how to save the data
    result = bilby.run_sampler(
        likelihood,
        prior,
        sampler="dynesty",
        outdir=outdir,
        label=settings["event_label"],
        nlive=settings.get("nlive", 2000),
        nact=settings.get("nact", 20),
        walks=settings.get("walks", 100),
        n_check_point=10000,
        check_point_plot=True,
        conversion_function=bilby.gw.conversion.generate_all_bbh_parameters,
        plot=False,
    )
    result.plot_corner()
