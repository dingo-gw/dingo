import numpy as np
import yaml
from scipy.interpolate import interp1d
from os.path import join
import matplotlib.pyplot as plt

from dingo.gw.noise.asd_dataset import ASDDataset
from dingo.gw.domains import build_domain

if __name__ == "__main__":
    wfd_settings_path = (
        "/Users/maxdax/Documents/Projects/GW-Inference/01_bns"
        "/local_runs/01_prototyping/01/waveform_dataset_settings_autocompleted.yaml"
    )
    asd_dataset_path = (
        "/Users/maxdax/Documents/Projects/GW-Inference/02_continuous_flows/"
        "local_runs/01_prototyping/01/asds_O1_fiducial.hdf5"
    )
    outdir = (
        "/Users/maxdax/Documents/Projects/GW-Inference/01_bns"
        "/local_runs/01_prototyping/01/"
    )
    decimation_method = ["inverse-asd-decimation", "psd-decimation"][0]
    with open(wfd_settings_path, "r") as f:
        wfd_settings = yaml.safe_load(f)

    mfd = build_domain(wfd_settings["domain"])
    ufd = mfd.base_domain
    asd_dataset = ASDDataset(file_name=asd_dataset_path)

    asd_dataset_decimated = {}

    for ifo, asds in asd_dataset.asds.items():
        asd_dataset_decimated[ifo] = np.zeros((len(asds), len(mfd)))
        for idx, asd in enumerate(asds):
            interp = interp1d(asd_dataset.domain(), asd)
            asd_ufd = interp(ufd())
            if decimation_method == "inverse-asd-decimation":
                asd_dataset_decimated[ifo][idx, :] = 1 / mfd.decimate(1 / asd_ufd)
            elif decimation_method == "psd-decimation":
                asd_dataset_decimated[ifo][idx, :] = (
                    1e-20 * mfd.decimate((asd_ufd * 1e20) ** 2) ** 0.5
                )
            else:
                raise NotImplementedError(
                    f"Unknown decimation method " f"{decimation_method}."
                )

    asd_dataset.asds = asd_dataset_decimated
    asd_dataset.settings["domain_dict"] = mfd.domain_dict
    asd_dataset.to_file(file_name=join(outdir, f"asd_dataset_{decimation_method}.hdf5"))

    asd_dataset = ASDDataset(
        file_name=join(outdir, f"asd_dataset_{decimation_method}.hdf5")
    )
    # test this
    asds_samples = asd_dataset.sample_random_asds()
    for ifo, asd in asds_samples.items():
        plt.plot(asd_dataset.domain(), asd, label=ifo)
    plt.yscale("log")
    plt.ylim((5e-24, 1e-20))
    plt.grid()
    plt.legend()
    plt.show()
