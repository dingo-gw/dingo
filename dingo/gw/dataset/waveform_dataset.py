import h5py
from typing import Dict, Union
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

    Once a waveform data set is in memory, the waveform data are consumed through a
    __getitem__() call, optionally applying a chain of transformations, which are classes
    that implement a __call__() method.
    """

    dataset_type = "waveform_dataset"

    def __init__(
        self,
        file_name: str | None = None,
        dictionary: dict | None = None,
        transform=None,
        precision: str | None = None,
        domain_update: dict | None = None,
        svd_size_update: int | None = None,
        leave_on_disk_keys: list | None = None,
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
        leave_on_disk_keys : dict
            If provided, not load the values for these keys into RAM. They are loaded lazily in __getitem__().
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
            leave_on_disk_keys=leave_on_disk_keys,
        )
        self.file_name = file_name

        if (
            self.settings is not None
        ):
            self.load_supplemental(domain_update, svd_size_update)
            self.svd_size_update = svd_size_update

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
        if self.precision is not None and self.complex_type is not None and self.real_type is not None:
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
        elif self.polarizations["h_cross"] is not None and self.polarizations["h_plus"] is not None:
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

        if "svd" in self.settings["compression"] and self.svd is not None:

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
                        if self.polarizations[k] is not None:
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

    def __getitem__(self, idx: Union[int, np.array]) -> Dict[str, Dict[str, Union[float, np.ndarray]]]:
        """
        Return a nested dictionary containing parameters and waveform polarizations
        for sample with index `idx`. If defined, a chain of transformations is applied to
        the waveform data.
        """
        if self.file_handle is None and self.leave_on_disk_keys is not None:
            # Open hdf5 file
            self.file_handle = h5py.File(self.file_name, "r")

        # Get parameters and data for idx
        if self.leave_on_disk_keys is None:
            # All data is in memory
            parameters = self.parameters.iloc[idx].to_dict()
            polarizations = {
                pol: waveforms[idx] for pol, waveforms in self.polarizations.items()
            }
        else:
            # Data or parameters are not in memory -> load them
            if "parameters" in self.leave_on_disk_keys and "h_cross" in self.leave_on_disk_keys and "h_plus" in self.leave_on_disk_keys:
                # Not working at the moment because standardization of parameters is calculated at training time
                data = recursive_hdf5_load(self.file_handle, keys=["parameters", "polarizations"], idx=idx)
                if self.svd is None:
                    # Apply domain update to set waveform to zero for f < f_min
                    polarizations = {
                        pol: self.domain.update_data(waveforms) for pol, waveforms in data["polarizations"].items()
                    }
                parameters = data["parameters"]
            elif "parameters" in self.leave_on_disk_keys:
                # Not working at the moment because standardization of parameters is calculated at training time
                parameters = recursive_hdf5_load(self.file_handle, keys=["parameters"], idx=idx)["parameters"]
                polarizations = {
                    pol: waveforms[idx] for pol, waveforms in self.polarizations.items()
                }
            elif "h_cross" in self.leave_on_disk_keys or "h_plus" in self.leave_on_disk_keys:
                polarizations = recursive_hdf5_load(self.file_handle, keys=["polarizations"], idx=idx)["polarizations"]
                if self.svd is None:
                    # Apply domain update to set waveform to zero for f < f_min
                    polarizations = {
                        pol: self.domain.update_data(waveforms) for pol, waveforms in polarizations.items()
                    }
                parameters = self.parameters.iloc[idx]
            else:
                raise ValueError(f"Unknown leave_on_disk_keys {self.leave_on_disk_keys}. "
                                 f"Cannot be loaded during __getitem__().")
            # Update precision
            if self.precision is not None and self.real_type is not None and self.complex_type is not None:
                parameters = parameters.astype(self.real_type, copy=False)
                for k, v in polarizations.items():
                        polarizations[k] = v.astype(self.complex_type, copy=False)
            if not isinstance(parameters, dict):
                parameters = parameters.to_dict()
            # Perform SVD size update on waveform
            if self.svd is not None and self.svd_size_update is not None:
                for k, v in polarizations.items():
                    if len(v.shape) == 1:
                        polarizations[k] = v[:self.svd_size_update]
                    else:
                        polarizations[k] = v[:, :self.svd_size_update]

        # Decompression transforms are assumed to apply only to the waveform,
        # and do not involve parameters.
        if self.decompression_transform is not None:
            polarizations = self.decompression_transform(polarizations)

        # Main transforms can depend also on parameters.
        data = {"parameters": parameters, "waveform": polarizations}
        if self.transform is not None:
            if isinstance(idx, int):
                data = self.transform(data)
            else:
                # TODO: replace by vectorized version of transforms,
                #  ideally, transforms can handle single data point and batch
                data = [self.transform(
                    {"parameters": {k: data["parameters"][k][j] for k in data["parameters"].keys()},
                     "waveform": {"h_cross": data["waveform"]["h_cross"][i], "h_plus": data["waveform"]["h_plus"][i]}}
                    ) for i, j in enumerate(idx)
                ]

        return data

    def __del__(self):
        if self.file_handle is not None:
            self.file_handle.close()

    def parameter_mean_std(self):
        mean = self.parameters.mean().to_dict()
        std = self.parameters.std().to_dict()
        return mean, std
