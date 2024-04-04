"""Create dingo ASD dataset based on txt PSD files.

python create_asd_dataset.py \
  --psd_filename "/fast/groups/dingo/03_binary_neutron_stars/00_data/01_GW170817/02_asds/GW170817_pePaper_PSD_<ifo>.txt" \
  --wfd_dir /fast/groups/dingo/03_binary_neutron_stars/01_wfd/03_generic_lowSpin/

python create_asd_dataset.py \
  --psd_filename "/fast/groups/dingo/03_binary_neutron_stars/00_data/02_GW190425/02_asds/GW190425_GWTC_PSD_<ifo>.txt" \
  --wfd_dir /fast/groups/dingo/03_binary_neutron_stars/01_wfd/03_generic_lowSpin/

python create_asd_dataset.py \
  --psd_filename "/fast/groups/dingo/03_binary_neutron_stars/00_data/01_GW170817/02_asds/GW170817_pePaper_PSD_<ifo>.txt" \
  --wfd_dir /fast/groups/dingo/03_binary_neutron_stars/01_wfd/03_generic_highSpin/

python create_asd_dataset.py \
  --psd_filename "/fast/groups/dingo/03_binary_neutron_stars/00_data/02_GW190425/02_asds/GW190425_GWTC_PSD_<ifo>.txt" \
  --wfd_dir /fast/groups/dingo/03_binary_neutron_stars/01_wfd/03_generic_highSpin/
"""
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.interpolate import interp1d
import yaml

from dingo.gw.domains import build_domain
from dingo.gw.noise.asd_dataset import ASDDataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--psd_filename",
    help="Filenames of PSD txt files, <ifo> is " "placeholder for ifo.",
)
parser.add_argument(
    "--wfd_dir",
    help="Directory with waveform dataset settings. ASDs " "saved in wfd_dir/asds.",
)
parser.add_argument("--plot", action="store_true")
args = parser.parse_args()

if "_<ifo>" not in args.psd_filename:
    raise ValueError()

wfd_settings = os.path.join(
    args.wfd_dir, "waveform_dataset_settings_autocompleted.yaml"
)
name = args.psd_filename.split("/")[-1].replace("_PSD_<ifo>.txt", "")
os.makedirs(os.path.join(args.wfd_dir, "asds"), exist_ok=True)
outname = os.path.join(args.wfd_dir, "asds", "asd_dataset_" + name + ".hdf5")

# load PSDs
psds = {}
frequencies = None
for ifo in ["H1", "L1", "V1", "K1"]:
    try:
        data = np.loadtxt(args.psd_filename.replace("<ifo>", ifo)).T
        if frequencies is None:
            frequencies = data[0]
        elif np.all(frequencies != data[0]):
            raise ValueError("Frequencies don't match.")
        psds[ifo] = data[1]
    except FileNotFoundError:
        pass
print(f"Found PSDs for ifos {psds.keys()}.")

# build target domain
with open(wfd_settings, "r") as f:
    wfd_settings = yaml.safe_load(f)
domain = build_domain(wfd_settings["domain"])
if frequencies[0] > domain.f_min:
    domain.update({"f_min": frequencies[0]})
if frequencies[-1] < domain.f_max:
    domain.update({"f_max": frequencies[-1]})

# Decimate to domain. We decimate the inverse ASDs, i.e.,
# ASD_decimated = 1 / decimate(1 / ASD), which is the appropriate operation for the ASD
# when we decimate the data *after* whitening (which best preserves the signal).
base_domain = domain.base_domain
asds_base_domain = {
    ifo: interp1d(frequencies, psd ** 0.5)(base_domain()[base_domain.min_idx :])
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
# asd_dataset.to_file(file_name=outname)

if args.plot:
    asd_dataset_loaded = asd_dataset  # ASDDataset(file_name=outname)
    for ifo, psd in psds.items():
        plt.plot(frequencies, psd ** 0.5, label=ifo)
        plt.plot(asd_dataset_loaded.domain(), asd_dataset_loaded.asds[ifo][0])
    plt.yscale("log")
    plt.show()
