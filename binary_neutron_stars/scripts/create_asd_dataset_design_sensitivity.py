import argparse
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import join
from scipy.interpolate import interp1d
import yaml
from bilby.gw.detector.psd import PowerSpectralDensity

from dingo.gw.domains import build_domain
from dingo.gw.noise.asd_dataset import ASDDataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--wfd_dir",
    help="Directory with waveform dataset settings. ASDs saved in wfd_dir/asds.",
)
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()


# load settings, build domain
wfd_settings = join(args.wfd_dir, "waveform_dataset_settings_autocompleted.yaml")
with open(wfd_settings, "r") as f:
    wfd_settings = yaml.safe_load(f)
domain = build_domain(wfd_settings["domain"])


# get design sensitivity psds
psd_ligo = PowerSpectralDensity.from_power_spectral_density_file(
    psd_file="aLIGO_ZERO_DET_high_P_psd.txt"
)
psd_virgo = PowerSpectralDensity.from_power_spectral_density_file(
    psd_file="AdV_psd.txt"
)
psds = {"H1": psd_ligo, "L1": psd_ligo, "V1": psd_virgo}


# Decimate to domain. We decimate the inverse ASDs, i.e.,
# ASD_decimated = 1 / decimate(1 / ASD), which is the appropriate operation for the ASD
# when we decimate the data *after* whitening (which best preserves the signal).
frequencies_base_domain = domain.base_domain()[domain.base_domain.min_idx :]
asds_base_domain = {
    ifo: interp1d(psd.frequency_array, psd.asd_array)(frequencies_base_domain)
    for ifo, psd in psds.items()
}
asds = {ifo: 1 / domain.decimate(1 / asd) for ifo, asd in asds_base_domain.items()}


# save dataset
asd_dataset = ASDDataset(
    dictionary={
        "settings": {"domain_dict": domain.domain_dict},
        "asds": {ifo: asd[np.newaxis, :] for ifo, asd in asds.items()},
        "gps_times": {ifo: np.array([-1]) for ifo in asds.keys()},
    }
)
asd_dataset.to_file(
    file_name=join(args.wfd_dir, "asds", "asd_dataset_design_sensitivity.hdf5")
)

if args.plot:
    asd_dir = join(args.wfd_dir, "asds")
    ifos = ["H1", "L1", "V1"]

    fig, axs = plt.subplots(len(ifos), 1)
    fig.set_size_inches(6, (len(ifos) * 4))
    filenames = [v for v in listdir(asd_dir) if v.endswith(".hdf5")]
    colors = ["black", "red", "blue"]
    for color, filename in zip(colors, filenames):
        asd_dataset = ASDDataset(file_name=join(asd_dir, filename))
        for ax, ifo in zip(axs, ifos):
            try:
                f = asd_dataset.domain()
                if True:
                    # absolute
                    ax.plot(f, asd_dataset.asds[ifo][0], label=filename, c=color)
                    ax.set_yscale("log")
                else:
                    # relative
                    y = asd_dataset.asds[ifo][0] / asds[ifo][:len(asd_dataset.asds[ifo][0])]
                    ax.plot(f, y, label=filename, c=color)
                    ax.set_ylim(0, 5)
            except KeyError:
                pass
    for ifo, ax in zip(ifos, axs):
        ax.set_title(ifo)
        ax.legend()
    plt.tight_layout()
    plt.savefig(join(asd_dir, "asd_comparison.pdf"))
    plt.show()
