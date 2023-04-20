import copy
from typing import Iterable

from dingo.gw.domains import build_domain
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
        this is [0, domain._f_max]. In practice one may want to train a network based on
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
        len_domain_original = len(self.domain)
        self.domain.update(domain_update)
        self.settings["domain"] = copy.deepcopy(self.domain.domain_dict)

        # truncate the dataset
        for ifo, asds in self.asds.items():

            # Is there a reason this check is needed? I would think that a dataset
            # should never be saved with this not matching.
            assert asds.shape[-1] == len_domain_original, (
                f"ASDs with shape {asds.shape[-1]} are not compatible"
                f"with the domain of length {len_domain_original}."
            )
            self.asds[ifo] = self.domain.update_data(
                asds,
                low_value=HIGH_ASD_VALUE,
            )

    def sample_random_asds(self):
        """
        Sample a random asd for each detector.
        Returns
        -------
        Dict with a random asd from the dataset for each detector.
        """
        return {k: v[np.random.choice(len(v), 1)[0]] for k, v in self.asds.items()}
