import copy
from typing import Iterable

from dingo.gw.domains import build_domain, Domain
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
        file_name: Optional[str] = None,
        dictionary: Optional[dict] = None,
        ifos: Optional[list] = None,
        precision: Optional[str] = None,
        domain_update: Optional[dict] = None,
        print_output: bool = True,
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
        print_output: bool
            Whether to write print statements to the console.
        """
        self.precision = precision
        super().__init__(
            file_name=file_name,
            dictionary=dictionary,
            data_keys=["asds", "gps_times", "asd_parameterizations"],
            print_output=print_output,
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
    def length_info(self) -> dict:
        """The number of asd samples per detector."""
        return {key: len(val) for key, val in self.asds.items()}

    @property
    def gps_info(self) -> dict:
        """Min/Max GPS time for each detector."""
        gps_info_dict = {}
        for key, val in self.gps_times.items():
            if not isinstance(val, Iterable):
                gps_info_dict[key] = val
            else:
                gps_info_dict[key] = (min(val), max(val))
        return gps_info_dict

    def update_domain(self, domain_update: dict):
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
                if self.print_output:
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


class MixedASDDatasetWrapper:
    """
    Wrapper for ASD datasets from multiple paths.
    """

    dataset_type = "mixed_asd_dataset"

    def __init__(
        self,
        asd_dataset_paths: list[str],
        probs: Optional[list[float]] = None,
        ifos: Optional[list[str]] = None,
        precision: Optional[str] = None,
        domain_update: Optional[dict[str, dict]] = None,
        print_output: bool = True,
    ):
        """
        Parameters
        ----------

        print_output: bool
            Whether to write print statements to the console.
        """
        self.asd_dataset_paths = asd_dataset_paths
        if probs is None:
            probs = (
                np.ones(len(self.asd_dataset_paths), dtype=float)
                * 1
                / len(self.asd_dataset_paths)
            )
        if not np.isclose(np.sum(np.array(probs)), 1.0, rtol=1e-6, atol=1e-12):
            raise ValueError(f"Probabilities probs {probs} does not sum to 1.")
        self.probs = np.array(probs)
        self.ifos = ifos
        # Load asd datasets
        if print_output:
            print("Loading multiple ASD datasets")
        self.asd_datasets = []
        for path in asd_dataset_paths:
            asd_dataset = ASDDataset(
                file_name=path,
                ifos=ifos,
                precision=precision,
                domain_update=domain_update,
                print_output=print_output,
            )
            self.asd_datasets.append(asd_dataset)

    @property
    def asds(self) -> dict[str, np.ndarray]:
        # Stack ASDs from individual ASD datasets
        asds_out = {}
        for a in self.asd_datasets:
            for k, v in a.asds.items():
                if k not in asds_out:
                    asds_out[k] = v
                else:
                    asds_out[k] = np.vstack([asds_out[k], v])
        return asds_out

    @property
    def gps_times(self) -> list[np.ndarray]:
        raise NotImplementedError()

    @property
    def asd_parameterizations(self) -> list[dict[str, np.ndarray]]:
        raise NotImplementedError()

    @property
    def domain(self) -> Domain:
        domain = self.asd_datasets[0].domain
        # Check that the domains of all sub-datasets are the same
        for i in range(1, len(self.asd_datasets)):
            if domain != self.asd_datasets[i].domain:
                raise ValueError(
                    f"Domain of ASD dataset 0: {domain} is not the same as the domain of"
                    f"ASD dataset {i}: {self.asd_datasets[i].domain}"
                )
        return domain

    @property
    def length_info(self) -> list[dict]:
        """The number of asd samples per detector per dataset."""
        raise NotImplementedError()

    @property
    def gps_info(self) -> list[dict]:
        raise NotImplementedError()

    def update_domain(self, domain_update: dict):
        for a in self.asd_datasets:
            a.update_domain(domain_update)

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
            Where the keys correspond to the detectors specified at initialization
            and the values  are arrays of shape (n, D) where D is the number of
            frequency bins and n is the number of ASDs requested. If n=None, then the
            function returns a single ASD for each detector, so the array is
            flattened to be shape D.
            If a dataset doesn't contain ASDs for a specific detector (e.g. O1),
            the ASD is set to HIGH_ASD_VALUE.
        """
        if n is None:
            idx_dataset = np.random.choice(
                len(self.asd_datasets), p=self.probs, replace=True
            )
            return self.asd_datasets[idx_dataset].sample_random_asds(n=None)
        else:
            # Sample dataset idx for each n
            idx_dataset = np.random.choice(
                len(self.asd_datasets), n, p=self.probs, replace=True
            )
            asds_out = {}
            for idx in np.unique(idx_dataset):
                num_samples = len(idx_dataset[idx_dataset == idx])
                # Load ASDs as batches of different lengths
                asds = self.asd_datasets[idx].sample_random_asds(n=num_samples)
                # Assemble batch
                sample_indices = np.argwhere(idx_dataset == idx)[..., 0]
                #
                if asds_out == {}:
                    asd_shape = asds[next(iter(asds))].shape[1:]
                    asds_out = {
                        i: HIGH_ASD_VALUE * np.ones([n, *asd_shape]) for i in self.ifos
                    }
                # Insert ASDs at correct positions
                for k in asds_out.keys():
                    if k in asds:
                        asds_out[k][sample_indices] = asds[k]

            return asds_out
