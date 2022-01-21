import os
from os.path import join
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import json
import h5py
from pathlib import Path
import argparse

from scipy.signal import tukey

from gwpy.timeseries import TimeSeries
import pycbc.psd

from dingo.gw.gwutils import (
    get_window,
    get_window_factor,
    save_dataset,
    load_dataset,
    recursive_hdf5_load,
)
from dingo.gw.domains import UniformFrequencyDomain, build_domain
from dingo.gw.transforms import WhitenAndScaleStrain
from dingo.gw.download_strain_data import download_event_data_in_FD

import yaml
import h5py
import ast

data_dir = "./datasets/strain_data/"


# time_segment = 8
# time_event = 1126259462.4
# time_buffer = 2
# num_segments_psd = 128
# T_psd = 1024.0

# detectors = ['H1', 'L1']
filename = "./datasets/strain_data/GW150914.hdf5"


def download_and_save_data_for_events(data_dir):
    with open(join(data_dir, "event_settings.yaml"), "r") as fp:
        event_settings = yaml.safe_load(fp)
    event_settings["window"]["T"] = event_settings["time_segment"]
    window = get_window(event_settings["window"])

    for name_event, time_event in event_settings["events"].items():
        print(f"Getting data for event {name_event}")
        data, domain = download_event_data_in_FD(
            event_settings["detectors"],
            time_event,
            event_settings["time_segment"],
            event_settings["time_buffer"],
            window,
            event_settings["num_segments_psd"],
        )
        filename = join(data_dir, f"{name_event}.hdf5")
        save_dataset(data, {"domain": domain.domain_dict}, filename)
        print("\n")

download_and_save_data_for_events(data_dir)

# f = h5py.File(filename, "r")
# data = recursive_hdf5_load(f)
# settings = ast.literal_eval(f.attrs["settings"])
# f.close()

data, settings = load_dataset(filename)
domain = build_domain(settings["domain"])

transforms = torchvision.transforms.Compose(
    [
        WhitenAndScaleStrain(domain.noise_std)
        # WhitenAndScaleStrain(1)
    ]
)

a = transforms(data)

ref_strains = h5py.File("./datasets/strain_data/old_data/strain_FD_whitened.hdf5", "r")

import matplotlib.pyplot as plt

plt.plot(a["waveform"]["H1"])
plt.plot(ref_strains["H1"][:])
plt.show()

ratio = ref_strains["H1"][200:] / a["waveform"]["H1"][200:8193]


d = {"waveform": {}, "asds": {}}
for det in detectors:
    print("Detector {:}:".format(det))

    d["waveform"][det] = download_strain_data_in_FD(det, t_event, T, T_buffer, window)
    d["asds"][det] = download_psd(det, t_event + T_buffer - T - T_psd, T, window) ** 0.5

    event_strain_FD_whitened[: int(f_min / event_strain_FD_whitened.delta_f)] = 0.0
    event_strain_FD_whitened = event_strain_FD_whitened[
        : int(f_max / event_strain_FD_whitened.delta_f) + 1
    ]


#
#     psd.save(join(event_dir, 'PSD_{:}.txt'.format(det)))
#
#     # store whitened strains and psds in dicts
#     whitened_FD_event_strains[det] = event_strain_FD_whitened
#     psds[det] = psd
#
# # Save whitened FD strain data
# print('Saving events strains to {:}'.format(event_dir / 'strain_FD_whitened.hdf5'))
# with h5py.File(event_dir / 'strain_FD_whitened.hdf5', 'w') as f:
#     for det in detectors:
#         strain_det = whitened_FD_event_strains[det]
#         f.create_dataset(det, data=strain_det)
#     f.create_dataset('sample_frequencies', data=np.array(strain_det.sample_frequencies.numpy()))
#
# # Save PSDs in numpy array
# print('Saving ASDs to {:}'.format(join(event_dir, 'ASDS.npy')))
# asds = {}
# for det in detectors:
#     asd = np.sqrt(np.array(psds[det][:8193]))
#     asds[det] = asd
# np.save(join(event_dir, 'ASDS.npy'), asds)


print("Done")
