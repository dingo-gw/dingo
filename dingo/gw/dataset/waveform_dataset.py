from typing import Dict, List, Optional, Union
import h5py
import numpy as np
import torch.utils.data
from torchvision.transforms import Compose

from dingo.core.dataset import DingoDataset, recursive_hdf5_load
from dingo.gw.SVD import SVDBasis, ApplySVD
from dingo.gw.domains import build_domain
from dingo.gw.transforms import WhitenFixedASD


class WaveformDataset(DingoDataset, torch.utils.data.Dataset):
    """This class stores a dataset of waveforms (polarizations) and corresponding
    parameters.

    It can load the dataset either from an HDF5 file or suitable dictionary.

    It is possible to either load the entire dataset into memory (leave_on_disk_keys=None),
    or to load the dataset on-demand (leave_on_disk_keys=['polarizations']).
    At the moment, it is only possible to load the polarizations on-demand since the
    standardization dict for all parameters in the dataset has to be computed at the
    beginning of training.
    # TODO: Compute the standardization dict when generating the waveform dataset, loading it at
    # the beginning of training, and adapting the code to work with leave_on_disk_keys=['parameters']

    The waveform data is consumed through a __getitem__() or __getitems__() call which optionally loads the
    polarizations and applies a chain of transformations, which are classes that implement a __call__() method.
    """

    dataset_type = "waveform_dataset"

    def __init__(
        self,
        file_name: Optional[str] = None,
        dictionary: Optional[dict] = None,
        transform=None,
        precision: Optional[str] = None,
        domain_update: Optional[dict] = None,
        svd_size_update: Optional[int] = None,
        wfd_keys_to_leave_on_disk: Optional[List] = None,
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
        wfd_keys_to_leave_on_disk : list
            If provided, the values for these keys are not loaded into RAM when initializing the
            waveform dataset. Instead, the values for these keys are loaded lazily in __getitem__().
        """
        self.domain = None
        self.transform = transform
        self.decompression_transform = None
        self.file_handle = None
        self.precision = precision
        if self.precision is not None:
            if self.precision == "single":
                self.complex_type = np.complex64
                self.real_type = np.float32
            elif self.precision == "double":
                self.complex_type = np.complex128
                self.real_type = np.float64
            else:
                raise TypeError(
                    'precision can only be changed to "single" or "double".'
                )
        else:
            self.real_type, self.complex_type = None, None
        super().__init__(
            file_name=file_name,
            dictionary=dictionary,
            data_keys=["parameters", "polarizations", "svd"],
            wfd_keys_to_leave_on_disk=wfd_keys_to_leave_on_disk,
        )
        self.file_name = file_name

        if self.settings is not None:
            self.load_supplemental(domain_update, svd_size_update)
            self.svd_size_update = svd_size_update

    def load_supplemental(
        self,
        domain_update: Optional[dict] = None,
        svd_size_update: Optional[int] = None,
    ):
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
        if (
            self.precision is not None
            and self.complex_type is not None
            and self.real_type is not None
        ):
            if self.parameters is not None:
                self.parameters = self.parameters.astype(self.real_type, copy=False)
            if (
                self.polarizations["h_cross"] is not None
                and self.polarizations["h_plus"] is not None
            ):
                for k, v in self.polarizations.items():
                    self.polarizations[k] = v.astype(self.complex_type, copy=False)

            # This should probably be moved to the SVDBasis class.
            if self.svd is not None:
                self.svd["V"] = self.svd["V"].astype(self.complex_type, copy=False)
                # For backward compatibility; in future, this will be there.
                if "s" in self.svd:
                    self.svd["s"] = self.svd["s"].astype(self.real_type, copy=False)

        if self.settings.get("compression", None) is not None:
            self.initialize_decompression(svd_size_update)

    def update_domain(self, domain_update: Optional[dict] = None):
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
        # itself (if it already has been loaded).
        if (
            self.settings.get("compression", None) is not None
            and "svd" in self.settings["compression"]
        ):
            self.svd["V"] = self.domain.update_data(self.svd["V"], axis=0)
        elif (
            self.polarizations["h_cross"] is not None
            and self.polarizations["h_plus"] is not None
        ):
            for k, v in self.polarizations.items():
                self.polarizations[k] = self.domain.update_data(v)

    def initialize_decompression(self, svd_size_update: Optional[int] = None):
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
                if self.polarizations is not None:
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

    def __getitem__(self, idx: int) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """
        Return a nested dictionary containing parameters and waveform polarizations
        for sample with index `idx`. If defined, a chain of transformations is applied to
        the waveform data.
        """
        return self.__getitems__([idx])[0]

    def __getitems__(
        self, batched_idx: list
    ) -> List[Dict[str, Dict[str, Union[float, np.ndarray]]]]:
        """
        Return a nested dictionary containing parameters and waveform polarizations
        for sample with index `idx`. If defined, a chain of transformations is applied to
        the waveform data.
        """
        if self.leave_on_disk_keys is not None and self.file_handle is None:
            # Open hdf5 file
            self.file_handle = h5py.File(self.file_name, "r")

        # Get parameters and data for idx
        if self.leave_on_disk_keys is None:
            parameters = {
                k: v if isinstance(v, float) else v.to_numpy()
                for k, v in self.parameters.iloc[batched_idx].items()
            }
            polarizations = {
                pol: waveforms[batched_idx]
                for pol, waveforms in self.polarizations.items()
            }
        else:
            # Data and/or parameters are not in memory -> load them
            if (
                "parameters" in self.leave_on_disk_keys
                and "h_cross" in self.leave_on_disk_keys
                and "h_plus" in self.leave_on_disk_keys
            ):
                raise ValueError(
                    "Loading parameters from disk is not implemented at the moment because parameter "
                    "standardization over the full dataset happens at the beginning of training. "
                    "Disable loading from disk or change leave_on_disk_keys to ['polarizations']."
                )
                # TODO: Adapt code to also load parameters from disk
                # * Compute parameter standardization when generating waveform dataset and save standardization_dict
                #   in dataset
                # * Load standardization dict at the beginning of training
                # * Adapt code to allow loading parameters from disk
            elif "parameters" in self.leave_on_disk_keys:
                raise ValueError(
                    "Loading parameters from disk is not implemented at the moment because parameter"
                    "standardization over the full dataset happens at the beginning of training."
                    "The standardization dict could be saved when generating the waveform dataset."
                )
                # TODO: Adapt code to also load parameters from disk (see comment above)
            elif (
                "h_cross" in self.leave_on_disk_keys
                or "h_plus" in self.leave_on_disk_keys
            ):
                polarizations = recursive_hdf5_load(
                    self.file_handle, keys=["polarizations"], idx=batched_idx
                )["polarizations"]
                if self.svd is None:
                    # Apply domain update to set waveform to zero for f < f_min
                    polarizations = {
                        pol: self.domain.update_data(waveforms)
                        for pol, waveforms in polarizations.items()
                    }
                parameters = {
                    k: v if isinstance(v, float) else v.to_numpy()
                    for k, v in self.parameters.iloc[batched_idx].items()
                }
            else:
                raise ValueError(
                    f"Unknown leave_on_disk_keys {self.leave_on_disk_keys}. "
                    f"Cannot be loaded during __getitem__()."
                )

            # Convert parameters to dict
            if not isinstance(parameters, dict):
                parameters = parameters.to_dict()
            # Update precision
            if (
                self.precision is not None
                and self.real_type is not None
                and self.complex_type is not None
            ):
                parameters = {
                    k: v.astype(self.real_type, copy=False)
                    for k, v in parameters.items()
                }
                polarizations = {
                    k: v.astype(self.complex_type, copy=False)
                    for k, v in polarizations.items()
                }
            # Perform SVD size update on waveform
            if self.svd is not None and self.svd_size_update is not None:
                for k, v in polarizations.items():
                    if len(v.shape) == 1:
                        polarizations[k] = v[: self.svd_size_update]
                    else:
                        polarizations[k] = v[:, : self.svd_size_update]

        # Decompression transforms are assumed to apply only to the waveform,
        # and do not involve parameters.
        if self.decompression_transform is not None:
            polarizations = self.decompression_transform(polarizations)

        # Main transforms can depend also on parameters.
        data = {"parameters": parameters, "waveform": polarizations}
        if self.transform is not None:
            data = self.transform(data)

        # Currently, the data is of shape [M, N, ...] with where M is the number
        # of arrays returned by the transform and N is the batch_size.  This
        # array is repackaged to group different indices of `M` into one sample,
        # resulting in data of shape [N, M, ...].  That is, data is of the form
        #
        # [arr1, ... arrM]
        #
        # where each arr is shape (N, ...).  Whereas the repackaged data is of form
        #
        # [[arr1[0, ...], ... arrM[0, ...]], ..., [arr1[N, ...], ... arrM[N, ...]]]
        #
        # which is a list of length N, where each element is an arr of shape (M, ...).
        # this is useful for collation
        repackaged_data = []
        if isinstance(data, dict):
            repackaged_data = [
                {k1: {k2: v2[j] for k2, v2 in v1.items()} for k1, v1 in data.items()}
                for j in range(len(batched_idx))
            ]
        elif isinstance(data, list):
            repackaged_data = [
                [data[i][j] for i in range(len(data))] for j in range(len(batched_idx))
            ]
        return repackaged_data

    def parameter_mean_std(self):
        mean = self.parameters.mean().to_dict()
        std = self.parameters.std().to_dict()
        return mean, std
