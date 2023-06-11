import numpy as np
import torch
from bilby.core.prior import PriorDict
from abc import ABC, abstractmethod

from dingo.gw.domains import Domain
from dingo.gw.transforms.waveform_transforms import HeterodynePhase


class GNPEBase(ABC):
    """
    A base class for Group Equivariant Neural Posterior Estimation [1].

    This implements GNPE for *approximate* equivariances. For exact equivariances,
    additional processing should be implemented within a subclass.

    [1]: https://arxiv.org/abs/2111.13139
    """

    def __init__(self, kernel_dict, operators):
        self.kernel = PriorDict(kernel_dict)
        self.operators = operators
        self.proxy_list = [k + "_proxy" for k in kernel_dict.keys()]
        self.context_parameters = self.proxy_list.copy()
        self.input_parameter_names = list(self.kernel.keys())

    @abstractmethod
    def __call__(self, input_sample):
        pass

    def sample_proxies(self, input_parameters):
        """
        Given input parameters, perturbs based on the
        kernel to produce "proxy" ("hatted") parameters, i.e., samples

            \hat g ~ p(\hat g | g).

        Typically the GNPE NDE will be conditioned on \hat g. Furthermore, these proxy
        parameters will be used to transform the data to simplify it.

        Parameters:
        -----------
        input_parameters : dict
            Initial parameter values to be perturbed. dict values can be either floats
            (for training) or torch Tensors (for inference).

        Returns
        -------
        A dict of proxy parameters.
        """
        proxies = {}
        for k in self.kernel:
            if k not in input_parameters:
                raise KeyError(
                    f"Input parameters are missing key {k} required for GNPE."
                )
            g = input_parameters[k]
            g_hat = self.perturb(g, k)
            proxies[k + "_proxy"] = g_hat
        return proxies

    def perturb(self, g, k):
        """
        Generate proxy variables based on initial parameter values.

        Parameters
        ----------
        g : Union[np.float64, float, torch.Tensor]
            Initial parameter values
        k : str
            Parameter name. This is used to identify the group binary operator.

        Returns
        -------
        Proxy variables in the same format as g.
        """
        # First we sample from the kernel, ensuring the correct data type,
        # and accounting for possible batching.
        #
        # Batching is implemented only for torch Tensors (expected at inference time),
        # whereas un-batched data in float form is expected during training.
        if type(g) == torch.Tensor:
            epsilon = self.kernel[k].sample(len(g))
            epsilon = torch.tensor(epsilon, dtype=g.dtype, device=g.device)
        elif type(g) == np.float64 or type(g) == float:
            epsilon = self.kernel[k].sample()
        else:
            raise NotImplementedError(f"Unsupported data type {type(g)}.")

        return self.multiply(g, epsilon, k)

    def multiply(self, a, b, k):
        op = self.operators[k]
        if op == "+":
            return a + b
        elif op == "x":
            return a * b
        else:
            raise NotImplementedError(
                f"Unsupported group multiplication operator: {op}"
            )

    def inverse(self, a, k):
        op = self.operators[k]
        if op == "+":
            return -a
        elif op == "x":
            return 1 / a
        else:
            raise NotImplementedError(
                f"Unsupported group multiplication operator: {op}"
            )


class GNPECoalescenceTimes(GNPEBase):
    """
    GNPE [1] Transformation for detector coalescence times.

    For each of the detector coalescence times, a proxy is generated by adding a
    perturbation epsilon from the GNPE kernel to the true detector time. This proxy is
    subtracted from the detector time, such that the overall time shift only amounts to
    -epsilon in training. This standardizes the input data to the inference network,
    since the applied time shifts are always restricted to the range of the kernel.

    To preserve information at inference time, conditioning of the inference network on
    the proxies is required. To that end, the proxies are stored in sample[
    'gnpe_proxies'].

    We can enforce an exact equivariance under global time translations, by subtracting
    one proxy (by convention: the first one, usually for H1 ifo) from all other
    proxies, and from the geocent time, see [1]. This is enabled with the flag
    exact_global_equivariance.

    Note that this transform does not modify the data itself. It only determines the
    amount by which to time-shift the data.

    [1]: arxiv.org/abs/2111.13139
    """

    def __init__(
        self, ifo_list, kernel, exact_global_equivariance=True, inference=False
    ):
        """
        Parameters
        ----------
        ifo_list : bilby.gw.detector.InterferometerList
            List of interferometers.
        kernel : str
            Defines a Bilby prior, to be used for all interferometers.
        exact_global_equivariance : bool = True
            Whether to impose the exact global time translation symmetry.
        inference : bool = False
            Whether to use inference or training mode.
        """
        self.ifo_time_labels = [ifo.name + "_time" for ifo in ifo_list]
        kernel_dict = {k: kernel for k in self.ifo_time_labels}
        operators = {k: "+" for k in self.ifo_time_labels}
        super().__init__(kernel_dict, operators)

        self.inference = inference
        self.exact_global_equivariance = exact_global_equivariance
        if self.exact_global_equivariance:
            # GNPE networks are conditioned on proxy variables relative to the
            # "preferred" proxy (typically H1). We give these a different name so that we
            # can keep track separately of the un-shifted proxies.
            del self.context_parameters[0]
            self.context_parameters = [p + "_relative" for p in self.context_parameters]

    def __call__(self, input_sample):
        sample = input_sample.copy()
        extrinsic_parameters = sample["extrinsic_parameters"].copy()

        # If proxies already exist, use them. Otherwise, sample them. Proxies may
        # already exist if provided by an unconditional initialization network when
        # attempting to recover the density from GNPE samples.

        # TODO: Reimplement in GNPEBase.sample_proxies().
        if set(self.proxy_list).issubset(extrinsic_parameters.keys()):
            new_parameters = {p: extrinsic_parameters[p] for p in self.proxy_list}
        else:
            new_parameters = self.sample_proxies(extrinsic_parameters)

        # If we are in training mode, we assume that the time shifting due to different
        # arrival times of the signal in individual detectors has not yet been applied
        # to the data; instead the arrival times are stored in extrinsic_parameters.
        # Hence we subtract off the proxy times from these arrival times, so that time
        # shifting of the data only has to be done once.

        if not self.inference:
            for k in self.ifo_time_labels:
                new_parameters[k] = (
                    -new_parameters[k + "_proxy"] + extrinsic_parameters[k]
                )
        # In inference mode, the data are only time shifted by minus the proxy.
        else:
            for k in self.ifo_time_labels:
                new_parameters[k] = -new_parameters[k + "_proxy"]

        # If we are imposing the global time shift symmetry, then we treat the first
        # proxy as "preferred", in the sense that it defines the global time shift.
        # This symmetry is enforced as follows:
        #
        #    1) Do not explicitly condition the model on the preferred proxy
        #    2) Subtract the preferred proxy from geocent_time (assumed to be a regression
        #    parameter). Note that this must be undone at inference time.
        #    3) Subtract the preferred proxy from the remaining proxies. These remaining
        #    proxies then define time shifts relative to the global time shift.
        #
        # Imposing the global time shift does not impact the transformation of the
        # data: we do not change the values of the true detector coalescence times
        # stored in extrinsic_parameters, only the proxies.

        if self.exact_global_equivariance:
            dt = new_parameters[self.ifo_time_labels[0] + "_proxy"]
            if not self.inference:
                if "geocent_time" not in extrinsic_parameters:
                    raise KeyError(
                        "geocent_time should be in extrinsic_parameters at "
                        "this point during training."
                    )
                new_parameters["geocent_time"] = (
                    extrinsic_parameters["geocent_time"] - dt
                )
            else:
                new_parameters["geocent_time"] = -dt
            for k in self.ifo_time_labels[1:]:
                new_parameters[k + "_proxy_relative"] = (
                    new_parameters[k + "_proxy"] - dt
                )

        extrinsic_parameters.update(new_parameters)
        sample["extrinsic_parameters"] = extrinsic_parameters
        return sample


class GNPEChirp(GNPEBase):
    """
    Relative binning / heterodyning GNPE transform, which factors out the overall chirp
    from the waveform. This is done based on the proxy parameters chirp_mass_proxy and
    optionally mass_ratio_proxy. These are defined as blurred version of the parameters
    chirp_mass and mass_ratio.

    At leading order, the data are transformed by dividing by a fiducial waveform of the
    form

        exp( - 1j * (3/128) * (pi G chirp_mass_proxy f / c**3)**(-5/3) ) ;

    see 2001.11412, eq. (7.2). This is the leading order chirp due to the emission of
    quadrupole radiation.

    At next to leading order, this transform also optionally implements 1PN corrections
    involving the mass ratio. We do not include any amplitude in the fiducial waveform,
    since at inference time this transform will be applied to noisy data. Multiplying
    the frequency-domain noise by a complex number of unit norm is allowed because it
    only changes the phase, not the overall amplitude, which would change the noise PSD.
    """

    def __init__(self, kernel, domain: Domain, order: int = 0, inference: bool = False):
        """
        Parameters
        ----------
        kernel : dict or str
            Defines a Bilby prior. If a dict, keys should include chirp_mass,
            and (possibly) mass_ratio.
        domain : Domain
            Only works for a FrequencyDomain at present.
        order : int
            Twice the post-Newtonian order for the expansion. Valid orders are 0 and 2.
        inference : bool = False
            Whether to use inference or training mode.
        """
        # We copy the kernel because the PriorDict constructor modifies the argument.
        kernel = kernel.copy()

        if order == 0:
            if "chirp_mass" not in kernel:
                raise KeyError(f"Kernel must include chirp_mass key.")
            if "mass_ratio" in kernel:
                print(
                    "Warning: mass_ratio kernel provided, but will be ignored for "
                    "order 0 GNPE."
                )
                kernel.pop("mass_ratio")
        elif order == 2:
            if "chirp_mass" not in kernel or "mass_ratio" not in kernel:
                raise KeyError(f"Kernel must include chirp_mass and mass_ratio keys.")
        else:
            raise ValueError(f"Order {order} invalid. Acceptable values are 0 and 2.")

        operators = {"chirp_mass": "x", "mass_ratio": "x"}
        super().__init__(kernel, operators)

        self.inference = inference
        self.phase_heterodyning_transform = HeterodynePhase(
            domain, order, inverse=False
        )

    def __call__(self, input_sample):
        sample = input_sample.copy()

        extrinsic_parameters = sample["extrinsic_parameters"].copy()

        # If proxies already exist, use them. Otherwise, sample them. Proxies may
        # already exist if provided by an unconditional initialization network when
        # attempting to recover the density from GNPE samples, or when using fixed
        # initialization parameters.
        # TODO: Reimplement in GNPEBase.sample_proxies().
        if set(self.proxy_list).issubset(extrinsic_parameters.keys()):
            proxies = {p: extrinsic_parameters[p] for p in self.proxy_list}
        else:
            # The relevant parameters could be in either the intrinsic or extrinsic
            # parameters list. At inference time, we put all GNPE parameters into the
            # extrinsic parameters list.
            proxies = self.sample_proxies(
                {**sample["parameters"], **sample["extrinsic_parameters"]}
            )
            extrinsic_parameters.update(proxies)
            sample["extrinsic_parameters"] = extrinsic_parameters

        # The only situation where we would expect to not have a waveform to transform
        # would be when calculating parameter standardizations, since we just want to
        # draw samples of the parameters at that point, and not prepare any data.
        if "waveform" in sample:
            sample["waveform"] = self.phase_heterodyning_transform(
                {
                    "waveform": sample["waveform"],
                    "parameters": {
                        "chirp_mass": proxies["chirp_mass_proxy"],
                        "mass_ratio": proxies.get("mass_ratio_proxy"),
                    },
                }
            )["waveform"]
            # import matplotlib.pyplot as plt
            # plt.plot(
            #     self.phase_heterodyning_transform.domain(),
            #     input_sample["waveform"]["h_cross"],
            # )
            # plt.plot(
            #     self.phase_heterodyning_transform.domain(),
            #     sample["waveform"]["h_cross"],
            # )
            # plt.show()
            # plt.xlim((0, 100))
            # plt.plot(input_sample["waveform"]["h_cross"])
            # plt.plot(sample["waveform"]["h_cross"])
            # plt.show()

        return sample
