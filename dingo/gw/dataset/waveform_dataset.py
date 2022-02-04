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
        """
        self.domain = None
        self.is_truncated = False
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
            self.load_supplemental()

    def load_supplemental(self):
        """Method called immediately after loading a dataset.

        Creates domain, updates dtypes, and initializes any decompression transform.
        """
        self.domain = build_domain(self.settings["domain"])

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

    def truncate_dataset_domain(self, new_range=None):
        """
        The waveform dataset provides waveform polarizations in a particular range. In
        uniform Frequency domain for instance, this range is [0, domain._f_max]. In
        practice one may want to apply data conditioning different to that of the
        dataset by specifying a different range, and truncating this dataset
        accordingly. That corresponds to truncating the likelihood integral.

        This method provides functionality for that. It truncates the dataset to the
        range specified by the domain, by calling domain.truncate_data. In uniform FD,
        this corresponds to truncating data in the range [0, domain._f_max] to the
        range [domain._f_min, domain._f_max].

        Before this truncation step, one may optionally modify the domain, to set a new
        range. This is done by domain.set_new_range(*new_range), which is called if
        new_range is not None.
        """
        if self.is_truncated:
            raise ValueError("Dataset is already truncated")
        # len_domain_original = len(self.domain)

        # Optionally set a new domain range.
        if new_range is not None:
            self.domain.set_new_range(*new_range)
            self.settings["domain"] = copy.deepcopy(self.domain.domain_dict)

        # Determine where the truncation must be applied. If the dataset is SVD
        # compressed, then truncate the SVD matrices. Otherwise, truncate the dataset
        # itself.
        if (
            self.settings["compression"] is not None
            and "svd" in self.settings["compression"]
        ):
            self.svd_V = self.domain.truncate_data(self.svd_V, axis=0)
            self.initialize_decompression()
        else:
            for k, v in self.polarizations.items():
                self.polarizations[k] = self.domain.truncate_data(v)

        # # truncate the dataset
        # if self._Vh is not None:
        #     assert self._Vh.shape[-1] == len_domain_original, \
        #         f'Compression matrix Vh with shape {self._Vh.shape} is not ' \
        #         f'compatible with the domain of length {len_domain_original}.'
        #     self._Vh = self.domain.truncate_data(
        #         self._Vh, allow_for_flexible_upper_bound=(new_range is not
        #                                                   None))
        # else:
        #     raise NotImplementedError('Truncation of the dataset is currently '
        #                               'only implemented for compressed '
        #                               'polarization data.')

        self.is_truncated = True
