import copy
from typing import Dict, Union
import numpy as np
import torch.utils.data
from torchvision.transforms import Compose

from dingo.core.dataset import DingoDataset
from dingo.gw.SVD import SVDBasis, ApplySVD
from dingo.gw.domains import build_domain
from dingo.gw.transforms import WhitenFixedASD


class WaveformDataset(DingoDataset, torch.utils.data.Dataset):
    """This class stores a dataset of waveforms (polarizations) and corresponding
    parameters.

    It can load the dataset either from an HDF5 file or suitable dictionary.

    Once a waveform data set is in memory, the waveform data are consumed through a
    __getitem__() call, optionally applying a chain of transformations, which are classes
    that implement a __call__() method.
    """

    dataset_type = "waveform_dataset"

    def __init__(
        self,
        file_name=None,
        dictionary=None,
        transform=None,
        precision=None,
        domain_update=None,
        svd_size_update=None,
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
        svd_size_update : int
            If provided, reduces the SVD size when decompressing (for speed).
        """
        self.domain = None
        self.transform = transform
        self.decompression_transform = None
        self.precision = precision
        super().__init__(
            file_name=file_name,
            dictionary=dictionary,
            data_keys=["parameters", "polarizations", "svd"],
        )

        if (
            self.parameters is not None
            and self.polarizations is not None
            and self.settings is not None
        ):
            self.load_supplemental(domain_update, svd_size_update)

    def load_supplemental(self, domain_update=None, svd_size_update=None):
        """Method called immediately after loading a dataset.

        Creates (and possibly updates) domain, updates dtypes, and initializes any
        decompression transform. Also zeros data below f_min, and truncates above f_max.

        Parameters
        ----------
        domain_update : dict
            If provided, update domain from existing domain using new settings.
        svd_size_update : int
            If provided, reduces the SVD size when decompressing (for speed).
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

            # This should probably be moved to the SVDBasis class.
            if self.svd is not None:
                self.svd["V"] = self.svd["V"].astype(complex_type, copy=False)
                # For backward compatibility; in future, this will be there.
                if "s" in self.svd:
                    self.svd["s"] = self.svd["s"].astype(real_type, copy=False)

        if self.settings.get("compression", None) is not None:
            self.initialize_decompression(svd_size_update)

    def update_domain(self, domain_update: dict = None):
        """
        Update the domain based on new configuration.

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

        # Determine where any domain adjustment must be applied. If the dataset is SVD
        # compressed, then adjust the SVD matrices. Otherwise, adjust the dataset
        # itself.
        if (
            self.settings.get("compression", None) is not None
            and "svd" in self.settings["compression"]
        ):
            self.svd["V"] = self.domain.update_data(self.svd["V"], axis=0)
        else:
            for k, v in self.polarizations.items():
                self.polarizations[k] = self.domain.update_data(v)

    def initialize_decompression(self, svd_size_update: int = None):
        """
        Sets up decompression transforms. These are applied to the raw dataset before
        self.transform. E.g., SVD decompression.

        Parameters
        ----------
        svd_size_update : int
            If provided, reduces the SVD size when decompressing (for speed).
        """
        decompression_transform_list = []

        # These transforms must be in reverse order compared to when dataset was
        # constructed.

        if "svd" in self.settings["compression"]:
            assert self.svd is not None

            # We allow the option to reduce the size of the SVD used for decompression,
            # since decompression is the costliest preprocessing operation. Be careful
            # when using this to not introduce a large mismatch.
            if svd_size_update is not None:
                if svd_size_update > self.svd["V"].shape[-1] or svd_size_update < 0:
                    raise ValueError(
                        f"Cannot truncate SVD from size "
                        f"{self.svd['V'].shape[-1]} to size "
                        f"{svd_size_update}."
                    )
                self.svd["V"] = self.svd["V"][:, :svd_size_update]
                self.svd["s"] = self.svd["s"][:svd_size_update]
                for k, v in self.polarizations.items():
                    self.polarizations[k] = v[:, :svd_size_update]

            svd_basis = SVDBasis(dictionary=self.svd)
            decompression_transform_list.append(ApplySVD(svd_basis, inverse=True))

        if "whitening" in self.settings["compression"]:
            decompression_transform_list.append(
                WhitenFixedASD(
                    self.domain,
                    asd_file=self.settings["compression"]["whitening"],
                    inverse=True,
                    precision=self.precision,
                )
            )

        self.decompression_transform = Compose(decompression_transform_list)

    def __len__(self):
        """The number of waveform samples."""
        return len(self.parameters)

    def __getitem__(self, idx) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """
        Return a nested dictionary containing parameters and waveform polarizations
        for sample with index `idx`. If defined, a chain of transformations is applied to
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
