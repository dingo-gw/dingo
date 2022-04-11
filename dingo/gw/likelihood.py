import numpy as np
from torchvision.transforms import Compose
from bilby import Likelihood
from bilby.gw.detector.networks import InterferometerList

from dingo.gw.waveform_generator import WaveformGenerator
from dingo.gw.domains import build_domain_from_model_metadata, FrequencyDomain
from dingo.gw.inference.data_preparation import get_domain_data
from dingo.gw.transforms import (
    ProjectOntoDetectors,
    GetDetectorTimes,
    WhitenAndScaleStrain,
)


class StationaryGaussianLikelihoodBBH(Likelihood):
    """
    Implements BBH likelihood for stationary, Gaussian noise.
    """

    def __init__(
        self,
        wfg_kwargs,
        domain,
        domain_data,
        t_ref=None,
    ):
        """
        Initialize the likelihood.

        Parameters
        ----------
        wfg_kwargs: dict
            Waveform generator parameters (at least approximant and f_ref).
        domain: dingo.gw.domains.Domain
            Domain object for strain data.
        domain_data: dict
            GW data. Contains strain data in domain_data["waveforms"] and asds in
            domain_data["asds"].
        t_ref: float
            Reference time; true geocent time for GW is t_ref + theta["geocent_time"].
        """
        super().__init__()

        # set up waveform generator
        self.wfg_kwargs = wfg_kwargs
        self.domain = domain
        if type(self.domain) is not FrequencyDomain:
            raise NotImplementedError(
                f"Likelihood implemented for FrequencyDomain, got {type(self.domain)}"
            )
        self.waveform_generator = WaveformGenerator(
            domain=self.domain,
            **self.wfg_kwargs,
        )

        # set GW event data
        self.t_ref = t_ref
        self.whitened_strains = {
            k: v / domain_data["asds"][k] for k, v in domain_data["waveform"].items()
        }
        self.asds = domain_data["asds"]
        if len(list(self.whitened_strains.values())[0]) != domain.max_idx + 1:
            raise ValueError("Strain data does not match domain.")

        # build transforms for detector projections
        self.ifo_list = InterferometerList(self.whitened_strains.keys())
        self.projection_transforms = Compose(
            [
                GetDetectorTimes(self.ifo_list, self.t_ref),
                ProjectOntoDetectors(self.ifo_list, self.domain, self.t_ref),
                WhitenAndScaleStrain(self.domain.noise_std),
            ]
        )

    def compute_gw_strain(self, theta):
        """
        Compute the GW strain for a given set of parameters theta.

        Step 1: generate polarizations h_plus and h_cross
        Step 2: project h_plus and h_cross onto detectors,
                whiten the signal, scale to account for window factor

        Parameters
        ----------
        theta: dict
            BBH parameters. Includes intrinsic parameters to be passed to waveform
            generator, and extrinsic parameters for detector projection.

        Returns
        -------
        gw_strain: dict
            GW strain for each detector.
        """
        theta_intrinsic, theta_extrinsic = split_off_extrinsic_parameters(theta)
        theta_intrinsic = {k: float(v) for k, v in theta_intrinsic.items()}

        # Step 1: generate polarizations h_plus and h_cross
        polarizations = self.waveform_generator.generate_hplus_hcross(theta_intrinsic)

        # Step 2: project h_plus and h_cross onto detectors
        sample = {
            "parameters": theta_intrinsic,
            "extrinsic_parameters": theta_extrinsic,
            "waveform": polarizations,
            "asds": self.asds,
        }
        sample = self.projection_transforms(sample)

        # import matplotlib.pyplot as plt
        # plt.xscale("log")
        # plt.plot(self.whitened_strains["H1"].real)
        # plt.plot(sample["waveform"]["H1"].real)
        # plt.show()

        return sample

    def log_likelihood(self, theta):
        """
        Compute the log likelihood for GW data (strains + asds) given parameters theta.

        Step 1: compute whitened GW strain h(theta) for parameters theta
        Step 2: subtract signal h from whitened strain data d, n = d - h
        Step 3: compute likelihood that n is Gaussian noise  with variance 1 on real and
                imaginary part individually

        Parameters
        ----------
        theta: dict
            BBH parameters. Includes intrinsic parameters to be passed to waveform
            generator, and extrinsic parameters for detector projection.

        Returns
        -------
        log_likelihood: float
            Log likelihood of the data whitened_strains given the parameters theta.
        """
        # Step 1: compute whitened GW strain h(theta) for parameters theta
        h = self.compute_gw_strain(theta)["waveform"]

        # Step 2: subtract signal h from whitened strain data d, n = d - h
        n = {k: v - h[k] for k, v in self.whitened_strains.items()}

        # Step 3: compute likelihood that n is Gaussian noise with variance 1 on real
        # and imaginary part individually
        log_likelihoods = {}
        for ifo, n_ifo in n.items():
            # truncate according to domain
            n_ifo = n_ifo[self.domain.min_idx :]
            # Compute likelihood. We assume that the noise is Gaussian and white (after
            # whitening), so the likelihood is given by N[0, 1](n). For a single bin i,
            # the log likelihood is this given by
            #
            #           log(N[0, 1](n[i])) = - log(sqrt(2) * pi) - 1/2. n[i] ** 2.
            #
            # To compute the log likelihood for a whole array, we sum over the array of
            # log likelihoods.
            # The considerations above hold for the real and imaginary part of the
            # noise individually, so we add both contributions.
            l_real = np.sum(-1 / 2.0 * n_ifo.real ** 2)
            l_imag = np.sum(-1 / 2.0 * n_ifo.imag ** 2)
            l_const =  - 2 * len(n_ifo) * np.log(np.sqrt(2) * np.pi)
            log_likelihoods[ifo] = l_real + l_imag + l_const

        return sum(log_likelihoods.values())

    def log_prob(self, *args, **kwargs):
        """
        Wraps log_likelihood method, required since downstream methods call
        distribution.log_prob, but bilby.Likelihood requires distribution.loq_likelihood.
        """
        return self.log_likelihood(*args, **kwargs)


def split_off_extrinsic_parameters(theta):
    """
    Split theta into intrinsic and extrinsic parameters.

    Parameters
    ----------
    theta: dict
        BBH parameters. Includes intrinsic parameters to be passed to waveform
        generator, and extrinsic parameters for detector projection.

    Returns
    -------
    theta_intrinsic: dict
        BBH intrinsic parameters.
    theta_extrinsic: dict
        BBH extrinsic parameters.
    """
    extrinsic_parameters = ["geocent_time", "luminosity_distance", "ra", "dec", "psi"]
    theta_intrinsic = {}
    theta_extrinsic = {}
    for k, v in theta.items():
        if k in extrinsic_parameters:
            theta_extrinsic[k] = v
        else:
            theta_intrinsic[k] = v
    # set fiducial values for time and distance
    theta_intrinsic["geocent_time"] = 0
    theta_intrinsic["luminosity_distance"] = 100
    return theta_intrinsic, theta_extrinsic


def main():
    import pandas as pd

    samples = (
        "/Users/maxdax/Documents/Projects/GW-Inference/dingo/datasets/dingo_samples"
        "/02_XPHM/dingo_samples_GW150914.pkl"
    )
    event_dataset = (
        "/Users/maxdax/Documents/Projects/GW-Inference/dingo/dingo-devel"
        "/tutorials/02_gwpe/datasets/strain_data/events_dataset.hdf5"
    )
    samples = pd.read_pickle(samples)
    metadata = samples.attrs
    wfg_kwargs = metadata["model"]["dataset_settings"]["waveform_generator"]
    domain = build_domain_from_model_metadata(metadata["model"])

    # get strain data
    domain_data = get_domain_data(
        metadata["model"], event_dataset=event_dataset, **metadata["event"]
    )

    # set up likelihood
    likelihood = StationaryGaussianLikelihoodBBH(
        wfg_kwargs, domain, domain_data, t_ref=metadata["event"]["time_event"]
    )

    from tqdm import tqdm
    log_likelihoods = []
    for idx in tqdm(range(1000)):
        theta = dict(samples.iloc[idx])
        try:
            l = likelihood.log_prob(theta)
        except:
            print(idx)
            l = float("nan")
        log_likelihoods.append(l)
    log_likelihoods = np.array(log_likelihoods)
    log_likelihoods = log_likelihoods[~np.isnan(log_likelihoods)]
    print(f"mean: {np.mean(log_likelihoods)}")
    print(f"std: {np.std(log_likelihoods)}")
    print(f"max: {np.max(log_likelihoods)}")
    print(f"min: {np.min(log_likelihoods)}")


if __name__ == "__main__":
    main()
