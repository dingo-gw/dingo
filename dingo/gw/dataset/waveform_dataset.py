import copy
from typing import Dict, Union
import numpy as np
import torch.utils.data
from torchvision.transforms import Compose

from dingo.core.dataset import DingoDataset
from dingo.gw.SVD import SVDBasis, UndoSVD
from dingo.gw.domains import build_domain


class WaveformDataset(DingoDataset, torch.utils.data.Dataset):
    """This class stores a dataset of waveforms (polarizations) and corresponding
    parameters.

    It can load the dataset either from an HDF5 file or suitable dictionary.

    Once a waveform data set is in memory, the waveform data are consumed through a
    __getitem__() call, optionally applying a chain of transformations, which are classes
    that implement a __call__() method.
    """

    def __init__(
        self,
        file_name=None,
        dictionary=None,
        transform=None,
        precision=None,
        domain_update=None,
    ):
        """
        For constructing, provide either file_name, or dictionary containing data and
        settings entries, or neither.

        Parameters
        ----------
        file_name : str
            HDF5 file containing a dataset
        dictionary : dict
            Contains settings and data entries. The dictionary keys should be
            'settings', 'parameters', and 'polarizations'.
        transform : Transform
            Transform to be applied to dataset samples when accessed through __getitem__
        precision : str ('single', 'double')
            If provided, changes precision of loaded dataset.
        domain_update : dict
            If provided, update domain from existing domain using new settings.
        """
        self.domain = None
        self.transform = transform
        self.decompression_transform = None
        self.precision = precision
        super().__init__(
            file_name=file_name,
            dictionary=dictionary,
            data_keys=["parameters", "polarizations", "svd_V"],
        )

        if (
            self.parameters is not None
            and self.polarizations is not None
            and self.settings is not None
        ):
            self.load_supplemental(domain_update)

    def load_supplemental(self, domain_update=None):
        """Method called immediately after loading a dataset.

        Creates (and possibly updates) domain, updates dtypes, and initializes any
        decompression transform. Also zeros data below f_min, and truncates above f_max.

        Parameters
        ----------
        domain_update : dict
            If provided, update domain from existing domain using new settings.
        """
        self.domain = build_domain(self.settings["domain"])

        # We always call update_domain() (even if domain_update is None) because we
        # want to be sure that the data are consistent with the saved settings. In
        # particular, this zeroes the waveforms for f < f_min.
        self.update_domain(domain_update)

        # Update dtypes if necessary
        if self.precision is not None:
            if self.precision == "single":
                complex_type = np.complex64
                real_type = np.float32
            elif self.precision == "double":
                complex_type = np.complex128
                real_type = np.float64
            else:
                raise TypeError(
                    'precision can only be changed to "single" or "double".'
                )
            self.parameters = self.parameters.astype(real_type, copy=False)
            for k, v in self.polarizations.items():
                self.polarizations[k] = v.astype(complex_type, copy=False)
            if self.svd_V is not None:
                self.svd_V = self.svd_V.astype(complex_type, copy=False)

        if self.settings["compression"] is not None:
            self.initialize_decompression()

    def update_domain(self, domain_update: dict = None):
        """
        Update the domain based on new configuration. Also adjust data arrays to match
        the new domain.

        The waveform dataset provides waveform polarizations in a particular domain. In
        Frequency domain, this is [0, domain._f_max]. Furthermore, data is set to 0 below
        domain._f_min. In practice one may want to train a network based on  slightly
        different domain settings, which corresponds to truncating the likelihood
        integral.

        This method provides functionality for that. It truncates and/or zeroes the
        dataset to the range specified by the domain, by calling domain.update_data.

        Parameters
        ----------
        domain_update : dict
            Settings dictionary. Must contain a subset of the keys contained in
            domain_dict.
        """
        if domain_update is not None:
            self.domain.update(domain_update)
            self.settings['domain'] = copy.deepcopy(self.domain.domain_dict)

        # Determine where any domain adjustment must be applied. If the dataset is SVD
        # compressed, then adjust the SVD matrices. Otherwise, adjust the dataset
        # itself.
        if (
                self.settings["compression"] is not None
                and "svd" in self.settings["compression"]
        ):
            self.svd_V = self.domain.update_data(self.svd_V, axis=0)
        else:
            for k, v in self.polarizations.items():
                self.polarizations[k] = self.domain.update_data(v)

    def initialize_decompression(self):
        """
        Sets up decompression transforms. These are applied to the raw dataset before
        self.transform. E.g., SVD decompression.
        """
        decompression_transform_list = []

        if "svd" in self.settings["compression"]:
            assert self.svd_V is not None
            svd_basis = SVDBasis()
            svd_basis.from_V(self.svd_V)
            decompression_transform_list.append(UndoSVD(svd_basis))

        self.decompression_transform = Compose(decompression_transform_list)

    def __len__(self):
        """The number of waveform samples."""
        return len(self.parameters)

    def __getitem__(self, idx) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """
        Return a nested dictionary containing parameters and waveform polarizations
        for sample with index `idx`. If defined, a chain of transformations are applied to
        the waveform data.
        """
        parameters = self.parameters.iloc[idx].to_dict()
        polarizations = {
            pol: waveforms[idx] for pol, waveforms in self.polarizations.items()
        }

        # Decompression transforms are assumed to apply only to the waveform,
        # and do not involve parameters.
        if self.decompression_transform is not None:
            polarizations = self.decompression_transform(polarizations)

        # Main transforms can depend also on parameters.
        data = {"parameters": parameters, "waveform": polarizations}
        if self.transform is not None:
            data = self.transform(data)
        return data

    def parameter_mean_std(self):
        mean = self.parameters.mean().to_dict()
        std = self.parameters.std().to_dict()
        return mean, std
