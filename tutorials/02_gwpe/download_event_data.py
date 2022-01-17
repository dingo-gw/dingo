import os
from os.path import join
import numpy as np
import matplotlib.pyplot as plt
import json
import h5py
from pathlib import Path
import argparse

from scipy.signal import tukey

from gwpy.timeseries import TimeSeries
import pycbc.psd

T = 8
t_event = 1126259462
T_buffer = 2

detectors = ['H1', 'L1']
roll_off = 0.4
alpha = 2 * roll_off / T
f_min = 20.0
f_max = 1024.0  # New f_max. Previously it was 2048 Hz.

whitened_FD_event_strains = {}
psds = {}
for det in detectors:
    print('Detector {:}:'.format(det))
    print('Downloading strain data for event.', end=' ')
    event_strain = TimeSeries.fetch_open_data(det, t_event + T_buffer - T,
                                              t_event + T_buffer, cache=False)
    # print('Done. \nDownloading strain data for PSD estimation.', end=' ')
    # psd_strain = TimeSeries.fetch_open_data(det, t_event + T_buffer - T - T_psd,
    #                                         t_event + T_buffer - T, cache=False)
    # print('Done.')

print('Done')