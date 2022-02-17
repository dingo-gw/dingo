import yaml

from bilby.gw.detector import InterferometerList

from dingo.gw.dataset.waveform_dataset import WaveformDataset
from dingo.gw.domains import build_domain
from dingo.gw.transforms import SampleExtrinsicParameters,\
    GetDetectorTimes, ProjectOntoDetectors, SampleNoiseASD, \
    WhitenAndScaleStrain, AddWhiteNoiseComplex, \
    SelectStandardizeRepackageParameters, RepackageStrainsAndASDS, UnpackDict, \
    GNPEDetectorTimes, GNPEChirpMass
from dingo.gw.noise_dataset import ASDDataset
from dingo.gw.prior import default_regression_parameters
from dingo.gw.gwutils import *

import numpy as np
import time

with open('./train_dir/train_settings.yaml', 'r') as fp:
    train_settings = yaml.safe_load(fp)

# build datasets
wfd = WaveformDataset(train_settings['waveform_dataset_path'])
asd_dataset = ASDDataset(
    train_settings['asd_dataset_path'],
    ifos=train_settings['transform_settings']['detectors'])
# truncate datasets
wfd.truncate_dataset_domain(
    train_settings['data_conditioning']['frequency_range'])
asd_dataset.truncate_dataset_domain(
    train_settings['data_conditioning']['frequency_range'])
# check compatibility of datasets
assert wfd.domain.domain_dict == asd_dataset.domain.domain_dict
# add window factor to domain
domain = build_domain(wfd.domain.domain_dict)
domain.window_factor = get_window_factor(
    train_settings['data_conditioning']['window_kwargs'])
assert domain.noise_std == 1.3692854996470123

extrinsic_prior_dict = get_extrinsic_prior_dict(
    train_settings['transform_settings']['extrinsic_prior'])
if train_settings['transform_settings']['regression_parameters'] == 'default':
    train_settings['transform_settings']['regression_parameters'] = default_regression_parameters
standardization_dict = get_standardization_dict(
    extrinsic_prior_dict, wfd,
    train_settings['transform_settings']['regression_parameters'])
ref_time = train_settings['transform_settings']['ref_time']
window_factor = get_window_factor(
    train_settings['data_conditioning']['window_kwargs'])
# build objects
ifo_list = InterferometerList(
    train_settings['transform_settings']['detectors'])

# build transforms
sample_extrinsic_parameters = SampleExtrinsicParameters(extrinsic_prior_dict)
get_detector_times = GetDetectorTimes(ifo_list, ref_time)
if 'gnpe_time_shifts' in train_settings['transform_settings']:
    d = train_settings['transform_settings']['gnpe_time_shifts']
    gnpe_time_shifts = GNPEDetectorTimes(
        ifo_list, d['kernel_kwargs'], d['exact_equiv'],
        std=standardization_dict['std']['geocent_time'])
if 'gnpe_chirp_mass' in train_settings['transform_settings']:
    d = train_settings['transform_settings']['gnpe_chirp_mass']
    gnpe_chirp_mass = GNPEChirpMass(
        domain.sample_frequencies_truncated,
        d['kernel_kwargs'],
        mean=standardization_dict['std']['chirp_mass'],
        std=standardization_dict['std']['chirp_mass'])
project_onto_detectors = ProjectOntoDetectors(ifo_list, domain, ref_time)
sample_noise_asd = SampleNoiseASD(asd_dataset)
whiten_scale_strain = WhitenAndScaleStrain(domain.noise_std)
add_noise = AddWhiteNoiseComplex()
select_standardize_repackage_params = SelectStandardizeRepackageParameters(
    standardization_dict)
repackage_strains_asds = RepackageStrainsAndASDS(
    train_settings['transform_settings']['detectors'])
unpack_dict = UnpackDict(selected_keys=['parameters', 'waveform'])

N = 10

t0 = time.time()
for idx in range(N):
    d0 = wfd[0]
print(f'{(time.time() - t0) / N:.3f} seconds')

t0 = time.time()
for idx in range(N):
    d1 = sample_extrinsic_parameters(d0)
    d2 = get_detector_times(d1)
    d_gnpe = gnpe_time_shifts(d2)
    d_gnpe_chirp_mass = gnpe_chirp_mass(d2)
    d3 = project_onto_detectors(d2)
    d4 = sample_noise_asd(d3)
    d5 = whiten_scale_strain(d4)
    d6 = add_noise(d5)
    d7 = select_standardize_repackage_params(d6)
    d8 = repackage_strains_asds(d7)
    theta, x = unpack_dict(d8)

print(f'{(time.time() - t0)/N:.3f} seconds')

import matplotlib.pyplot as plt

plt.plot(d2['waveform']['h_cross'].real, label='original')
plt.plot(d_gnpe_chirp_mass['waveform']['h_cross'].real, label='gnpe chirp mass')
plt.legend()
plt.show()

plt.plot(np.abs(d2['waveform']['h_cross']), label='original')
plt.plot(np.abs(d_gnpe_chirp_mass['waveform']['h_cross']), label='gnpe chirp mass')
plt.legend()
plt.show()

a = np.random.rand(500)
Vh = np.random.rand(500, 8033)

# (500,)(500, 8033)

import matplotlib.pyplot as plt

d2['extrinsic_parameters']['H1_time'] = 0
a = project_onto_detectors(d2)
d2['extrinsic_parameters']['H1_time'] = 0.050
b = project_onto_detectors(d2)

fig, ax = plt.subplots()
ax.axis('off')
plt.xlim((0,2000))
plt.plot(a['waveform']['H1'])
plt.savefig('waveform_shift_0.pdf')

fig, ax = plt.subplots()
ax.axis('off')
plt.xlim((0,2000))
plt.plot(b['waveform']['H1'])
plt.savefig('waveform_shift_50ms.pdf')


plt.plot(wfd[0]['waveform']['h_cross'].real /
         d3['parameters']['luminosity_distance'] * 100)
plt.plot(d3['waveform']['H1'].real)
plt.show()

plt.plot(d5['waveform']['H1'].real)
plt.show()

plt.yscale('log')
plt.plot(d5['asds']['H1'])
plt.show()

plt.title('strain.real in H1')
plt.xscale('log')
plt.plot(domain.sample_frequencies_truncated,
         d6['waveform']['H1'].real, label='noisy waveform')
plt.plot(domain.sample_frequencies_truncated,
         d5['waveform']['H1'].real, label='pure waveform')
plt.legend()
plt.show()

ref_data = np.load('train_dir/waveform_data.npy', allow_pickle=True).item()
sample_in = {'parameters': ref_data['intrinsic_parameters'],
             'waveform': {'h_cross': ref_data['hc'],
                          'h_plus': ref_data['hp']},
             'extrinsic_parameters': ref_data['extrinsic_parameters']}

# check that we packaged the polarizations correctly
assert sample_in.keys() == d1.keys()
for k in sample_in.keys():
    assert sample_in[k].keys() == d1[k].keys()

sample_out = get_detector_times(sample_in)
sample_out = project_onto_detectors(sample_out)

ifo_name = 'L1'

new = sample_out['waveform'][ifo_name].real
ref = ref_data['h_d_unwhitened'][ifo_name].real
deviation = new - ref

plt.xlim((0,250))
plt.xlabel('f in Hz')
plt.title(f'strain.real in {ifo_name}')
plt.plot(domain(), ref.real, '.', label='research code')
plt.plot(domain(), new.real, label='dingo code')
# plt.plot(domain(), deviation, label='deviation')
plt.legend()
plt.show()

plt.xlim((0,1024))
plt.xlabel('f in Hz')
plt.title('Deviation between research code and Dingo strain (H1)')
plt.plot(deviation)
plt.show()

print('done')