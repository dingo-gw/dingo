import copy
from typing import Iterable, Optional

from dingo.gw.domains import build_domain, UniformFrequencyDomain
from dingo.gw.domains.base_frequency_domain import BaseFrequencyDomain
from dingo.gw.gwutils import *
from dingo.gw.dataset import DingoDataset

HIGH_ASD_VALUE = 1.0


class ASDDataset(DingoDataset):
    """
    Dataset of amplitude spectral densities (ASDs). The ASDs are typically
    used for whitening strain data, and additionally passed as context to the
    neural density estimator.
    """

    dataset_type = "asd_dataset"

    def __init__(
        self,
        file_name=None,
        dictionary=None,
        ifos=None,
        precision=None,
        domain_update=None,
    ):
        """
        Parameters
        ----------
        file_name : str
            HDF5 file containing a dataset
        dictionary : dict
            Contains settings and data entries. The dictionary keys should be
            'settings', 'asds', and 'gps_times'.
        ifos : List[str]
            List of detectors used for dataset, e.g. ['H1', 'L1'].
            If not set, all available ones in the dataset are used.
        precision : str ('single', 'double')
            If provided, changes precision of loaded dataset.
        domain_update : dict
            If provided, update domain from existing domain using new settings.
        """
        self.precision = precision
        super().__init__(
            file_name=file_name,
            dictionary=dictionary,
            data_keys=["asds", "gps_times", "asd_parameterizations"],
        )

        if ifos is not None:
            for ifo in list(self.asds.keys()):
                if ifo not in ifos:
                    self.asds.pop(ifo)
                    self.gps_times.pop(ifo)

        self.domain = build_domain(self.settings["domain_dict"])
        if not check_domain_compatibility(self.asds, self.domain):
            raise ValueError("ASDs in dataset not compatible with domain.")
        if domain_update is not None:
            self.update_domain(domain_update)

        # Update dtypes if necessary
        if self.precision is not None:
            if self.precision == "single":
                for ifo, asd in self.asds.items():
                    self.asds[ifo] = asd.astype(np.float32, copy=False)
            elif self.precision == "double":
                for ifo, asd in self.asds.items():
                    self.asds[ifo] = asd.astype(np.float64, copy=False)
            else:
                raise TypeError(
                    'precision can only be changed to "single" or "double".'
                )

    @property
    def length_info(self):
        """The number of asd samples per detector."""
        return {key: len(val) for key, val in self.asds.items()}

    @property
    def gps_info(self):
        """Min/Max GPS time for each detector."""
        gps_info_dict = {}
        for key, val in self.gps_times.items():
            if not isinstance(val, Iterable):
                gps_info_dict[key] = val
            else:
                gps_info_dict[key] = (min(val), max(val))
        return gps_info_dict

    def update_domain(self, domain_update):
        """
        Update the domain based on new configuration. Also adjust data arrays to match
        the new domain.

        The ASD dataset provides ASDs in a particular domain. In Frequency domain,
        this is [0, domain.f_max]. In practice one may want to train a network based on
        slightly different domain settings, which corresponds to truncating the likelihood
        integral.

        This method provides functionality for that. It truncates the data below a
        new f_max, and sets the ASD below f_min to a large but finite value.

        Parameters
        ----------
        domain_update : dict
            Settings dictionary. Must contain a subset of the keys contained in
            domain_dict.
        """
        # Note that we require domain_update to have a type specified, even if
        # unchanged from the original domain. This reduces risks of errors.
        if self.domain.domain_dict["type"] == domain_update["type"]:
            self.domain.update(domain_update)
            self.settings["domain"] = copy.deepcopy(self.domain.domain_dict)

            # truncate the dataset
            for ifo, asds in self.asds.items():
                self.asds[ifo] = self.domain.update_data(
                    asds,
                    low_value=HIGH_ASD_VALUE,
                )

        elif (
            self.domain.domain_dict["type"] == "UniformFrequencyDomain"
            and domain_update["type"] == "MultibandedFrequencyDomain"
        ):
            print("Updating ASD dataset to MultibandedFrequencyDomain.")
            asd_dataset_decimated = {}
            mfd = build_domain(domain_update)
            ufd = mfd.base_domain
            if not check_domain_compatibility(self.asds, ufd):
                # If the ASD length is not compatible with the new base UFD,
                # first truncate it.
                print(
                    f"  Truncating first to new base UniformFrequencyDomain: f_max "
                    f"{self.domain.f_max} Hz -> {ufd.f_max} Hz"
                )
                self.domain.update(ufd.domain_dict)  # Additional compatibility check.
                for ifo, asds in self.asds.items():
                    self.asds[ifo] = self.domain.update_data(
                        asds,
                        low_value=HIGH_ASD_VALUE,
                    )

            decimation_method = "inverse-asd-decimation"

            for ifo, asds in self.asds.items():
                asd_dataset_decimated[ifo] = np.zeros((len(asds), len(mfd)))
                for idx, asd in enumerate(asds):
                    if decimation_method == "inverse-asd-decimation":
                        asd_dataset_decimated[ifo][idx, :] = 1 / mfd.decimate(1 / asd)
                    elif decimation_method == "psd-decimation":
                        asd_dataset_decimated[ifo][idx, :] = (
                            1e-20 * mfd.decimate((asd * 1e20) ** 2) ** 0.5
                        )
                    else:
                        raise NotImplementedError(
                            f"Unknown decimation method " f"{decimation_method}."
                        )

            self.asds = asd_dataset_decimated
            self.settings["domain_dict"] = mfd.domain_dict
            self.domain = mfd

        else:
            raise NotImplementedError(
                f"Cannot update ASD domain type "
                f"{self.domain.domain_dict['type']} to {domain_update['type']}"
            )

    def sample_random_asds(self, n: Optional[int] = None) -> dict[str, np.ndarray]:
        """
        Sample n random ASDs for each detector.

        Parameters
        ----------
        n : int
            Number of asds to sample

        Returns
        -------
        dict[str, np.ndarray]
            Where the keys correspond to the detectors and the values
            are arrays of shape (n, D) where D is the number of frequency bins
            and n is the number of ASDs requested. If n=None, then the
            function returns a single ASD for each detector, so the array is
            flattened to be shape D
        """
        if n is None:
            return {k: v[np.random.choice(len(v), 1)[0]] for k, v in self.asds.items()}
        else:
            return {k: v[np.random.choice(len(v), n)] for k, v in self.asds.items()}


def check_domain_compatibility(data: dict, domain: BaseFrequencyDomain) -> bool:
    for v in data.values():
        if not domain.check_data_compatibility(v):
            return False
    return True
