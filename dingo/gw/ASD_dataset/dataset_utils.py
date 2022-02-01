import os
import numpy as np
import requests
from typing import Dict, List
from functools import partial
from tqdm import tqdm
from multiprocessing import Pool
from dingo.gw.download_strain_data import get_window
from gwpy.timeseries import TimeSeries
from dataset_utils import *
from dingo.gw.domains import build_domain
import pycbc
import h5py
from os.path import join
from io import StringIO
from tqdm import trange
from dingo.core.dataset import recursive_hdf5_save
"""
Contains links for PSD segment lists with quality label BURST_CAT2 from the Gravitationa Wave Open Science Center.
Some events are split up into multiple chunks such that there are multiple URLs for one observing run
"""
URL_DIRECTORY = {
    "O1_L1": ["https://www.gw-openscience.org/timeline/segments/O1/L1_BURST_CAT2/1126051217/11203200/"],
    "O1_H1": ["https://www.gw-openscience.org/timeline/segments/O1/H1_BURST_CAT2/1126051217/11203200/"],

    "O2_L1": ["https://www.gw-openscience.org/timeline/segments/O2_16KHZ_R1/L1_BURST_CAT2/1164556817/23176801/"],
    "O2_H1": ["https://www.gw-openscience.org/timeline/segments/O2_16KHZ_R1/H1_BURST_CAT2/1164556817/23176801/"],
    "O2_V1": ["https://www.gw-openscience.org/timeline/segments/O2_16KHZ_R1/V1_BURST_CAT2/1164556817/23176801/"],

    "O3_L1": ["https://www.gw-openscience.org/timeline/segments/O3a_16KHZ_R1/L1_BURST_CAT2/1238166018/15811200/",
              "https://www.gw-openscience.org/timeline/segments/O3b_16KHZ_R1/L1_BURST_CAT2/1256655618/12708000/"],
    "O3_H1": ["https://www.gw-openscience.org/timeline/segments/O3a_16KHZ_R1/H1_BURST_CAT2/1238166018/15811200/",
              "https://www.gw-openscience.org/timeline/segments/O3b_16KHZ_R1/H1_BURST_CAT2/1256655618/12708000/"],
    "O3_V1": ["https://www.gw-openscience.org/timeline/segments/O3a_16KHZ_R1/V1_BURST_CAT2/1238166018/15811200/",
              "https://www.gw-openscience.org/timeline/segments/O3b_16KHZ_R1/V1_BURST_CAT2/1256655618/12708000/"]
}

def get_segment(segs, seg_start, t_lower, t_upper):
    """
    Parameters
    ----------
    segs : Tuple[int, int]
        Contains the start- and end gps_times that have been fetched from the GWOSC website
    seg_start : int
        Index from where to start checking if the segment is contained in the upper and lower bound
    t_lower : int
        Lower gps_time bound
    t_upper : int
        Upper gps_time bound

    Returns
    -------
    Index of a single segment and a flag whether the segment is contained in the upper and lower gps_time bound
    """
    for ind in range(seg_start, len(segs)):
        if segs[ind][1] >= t_lower:
            if segs[ind][0] <= t_lower and segs[ind][1] >= t_upper:
                return ind, True
            else:
                return ind, False
    return None, None

def get_valid_segments(segs, T_PSD, delta_T):
    """
    Given the segments `segs` and the time constraints `T_PSD`, `delta_T`, return all segments
    that can be used to estimate a PSD

    Parameters
    ----------
    segs : Tuple[int, int]
        Contains the start- and end gps_times that have been fetched from the GWOSC website
    T_PSD : str
        number of seconds used to estimate PSD
    delta_T : str
        number of seconds between two adjacent PSDs

    Returns
    -------
    All segments that can be used to estimate a PSD
    """
    t_start = segs[0][0]
    time_total = segs[-1][1] - segs[0][0]
    num_PSD_max = int(time_total / (T_PSD + delta_T))
    segment = 0
    PSD_segs_valid = []

    for ind in range(num_PSD_max):
        t_lower = ind * (T_PSD + delta_T) + t_start
        t_upper = t_lower + T_PSD
        segment, flag = get_segment(segs, segment, t_lower, t_upper)

        if flag:
            PSD_segs_valid.append((t_lower, t_upper))

    return PSD_segs_valid

def get_path_raw_data(data_dir, run, detector, T_PSD=1024, delta_T=1024):
    """
    Return the directory where the PSD data is to be stored
    Parameters
    ----------
    data_dir : str
        Path to the directory where the PSD dataset will be stored
    run : str
        Observing run that is used for the PSD dataset generation
    detector : str
        Detector that is used for the PSD dataset generation
    T_PSD : str
        number of seconds used to estimate PSD
    delta_T : str
        number of seconds between two adjacent PSDs

    Returns
    -------
    the path where the data is stored
    """
    return os.path.join(data_dir, 'tmp', 'raw_PSDs', run, detector, str(T_PSD) + '_' + str(delta_T))

def download_and_estimate_PSDs(data_dir: str, run: str, detector: str, settings: dict, verbose=False):
    """
    Download segment lists from the official GWOSC website that have the BURST_CAT_2 quality label. A .npy file
    is created for every PSD that will be in the final dataset. These are stored in data_dir/tmp and may be removed
    once the final dataset has been created.
    
    Parameters
    ----------
    data_dir : str
        Path to the directory where the PSD dataset will be stored
    run : str
        Observing run that is used for the PSD dataset generation
    detector : str
        Detector that is used for the PSD dataset generation
    settings : dict
        Dictionary of settings that are used for the dataset generation
    verbose : bool
        If true, there will be a progress bar indicating 

    -------

    """""

    filename = f'{run}_{detector}_BURST_CAT2.txt'
    key = run + "_" + detector
    urls = URL_DIRECTORY[key]
    segment_path = join(data_dir, 'tmp', 'segment_lists')

    starts, stops, durations = [], [], []
    for url in urls:
        r = requests.get(url, allow_redirects=True)
        c = StringIO(r.content.decode("utf-8"))
        starts_seg, stops_seg, durations_seg = np.loadtxt(c, dtype='int', unpack=True)
        starts = np.hstack([starts, starts_seg])
        stops = np.hstack([stops, stops_seg])
        durations = np.hstack([durations, durations_seg])

    T_PSD = settings['T_PSD']
    delta_T = settings['delta_T']

    f_s = settings['window']['f_s']
    roll_off = settings['window']['roll_off']
    T = settings['window']['T']
    w = get_window(settings['window'])

    path_raw_psds = get_path_raw_data(data_dir, run, detector, T_PSD, delta_T)
    os.makedirs(path_raw_psds, exist_ok=True)

    valid_segments = get_valid_segments(list(zip(starts, stops)), T_PSD=T_PSD, delta_T=delta_T)
    num_psds_max = len(valid_segments) if settings['num_psds_max'] <= 0 else settings['num_psds_max']
    valid_segments = valid_segments[:num_psds_max]
    print(f'Fetching data and computing Welch\'s estimate of {num_psds_max} valid segments:\n')

    for index, (start, end) in enumerate(tqdm(valid_segments, disable=not verbose)):
        filename = join(path_raw_psds, 'psd_{:05d}.npy'.format(index))

        if not os.path.exists(filename):
            psd = TimeSeries.fetch_open_data(detector, start, end, cache=False)
            assert f_s == len(psd) / T_PSD, 'Unexpected sampling frequency. A different one is used for Tukey window.'
            psd = psd.to_pycbc()
            psd_final = pycbc.psd.estimate.welch(psd, seg_len=int(T * f_s), seg_stride=int(T * f_s), window=w,
                                                 avg_method='median')
            np.save(filename,
                    {'detector': detector, 'segment': (index, start, end), 'time': (start, end), 'psd': psd_final,
                     'tukey window': {'f_s': f_s, 'roll_off': roll_off, 'T': T}})


def create_dataset_from_files(data_dir: str, run: str, detectors: List[str], settings: dict):

    """
    Creates a .hdf5 ASD datset file for an observing run using the estimated detector PSDs.

    Parameters
    ----------
    data_dir : str
        Path to the directory where the PSD dataset will be stored
    run : str
        Observing run that is used for the ASD dataset generation
    detectors : List[str]
        Detector data that is used for the ASD dataset generation
    settings : dict
        Dictionary of settings that are used for the dataset generation
    -------
    """

    f_min = settings['f_min']
    f_max = settings['f_max']
    T_PSD = settings['T_PSD']
    delta_T = settings['delta_T']

    domain_settings = {}

    save_dict = {}
    asds_dict = {}
    gps_times_dict = {}

    for ifo in detectors:

        path_raw_psds = get_path_raw_data(data_dir, run, ifo, T_PSD, delta_T)
        raw_PSDs = [el for el in os.listdir(path_raw_psds) if el.endswith('.npy')]

        psd = np.load(join(path_raw_psds, raw_PSDs[0]), allow_pickle=True).item()
        freqs = np.array(psd['psd'].sample_frequencies)


        delta_f =  freqs[1] - freqs[0]
        domain = build_domain({'type': 'FrequencyDomain',
                               'f_min': f_min, 'f_max': f_max,
                               'delta_f': delta_f})
        domain_settings['domain_dict'] = domain.domain_dict
        ind_min, ind_max = np.where(freqs == f_min)[0].item(), np.where(freqs == f_max)[0].item()
        assert ind_min is not None, 'f_min is not in sample frequencies of the PSD'
        assert ind_max is not None, 'f_max is not in sample frequencies of the PSD'

        Nf = ind_max - ind_min + 1
        asds = np.zeros((len(raw_PSDs), Nf))
        times = np.zeros(len(raw_PSDs))

        for ind, psd_name in enumerate(raw_PSDs):
            psd = np.load(join(path_raw_psds, psd_name), allow_pickle=True).item()
            asds[ind, :] = np.sqrt(psd['psd'][ind_min:ind_max + 1])
            times[ind] = psd['time'][0]

        asds_dict[ifo] = asds
        gps_times_dict[ifo] = times

    save_dict['asds'] = asds_dict
    save_dict['gps_times'] = gps_times_dict

    f = h5py.File(join(data_dir, f'asds_{run}_test.hdf5'), "w")
    recursive_hdf5_save(f, save_dict)
    f.attrs["settings"] = str(domain_settings)
    f.close()