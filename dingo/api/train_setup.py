import torchvision
from torch.utils.data import DataLoader
from bilby.gw.detector import InterferometerList

from dingo.gw.waveform_dataset import WaveformDataset
from dingo.gw.domains import build_domain
from dingo.gw.transforms import SampleExtrinsicParameters,\
    GetDetectorTimes, ProjectOntoDetectors, SampleNoiseASD, \
    WhitenAndScaleStrain, AddWhiteNoiseComplex, \
    SelectStandardizeRepackageParameters, RepackageStrainsAndASDS, \
    UnpackDict, GNPEDetectorTimes
from dingo.gw.noise_dataset import ASDDataset
from dingo.gw.prior_split import default_params
from dingo.gw.gwutils import *
from dingo.core.utils import *

def build_dataset(train_settings):
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

    extrinsic_prior_dict = get_extrinsic_prior_dict(
        train_settings['transform_settings']['extrinsic_prior'])
    if train_settings['transform_settings']['selected_parameters'] == 'default':
        train_settings['transform_settings']['selected_parameters'] = default_params
    standardization_dict = get_standardization_dict(
        extrinsic_prior_dict, wfd,
        train_settings['transform_settings']['selected_parameters'])
    ref_time = train_settings['transform_settings']['ref_time']
    window_factor = get_window_factor(
        train_settings['data_conditioning']['window_kwargs'])
    # build objects
    ifo_list = InterferometerList(
        train_settings['transform_settings']['detectors'])

    # build transforms
    gnpe_proxy_dim = 0
    transforms = []
    transforms.append(SampleExtrinsicParameters(extrinsic_prior_dict))
    transforms.append(GetDetectorTimes(ifo_list, ref_time))
    if 'gnpe_time_shifts' in train_settings['transform_settings']:
        d = train_settings['transform_settings']['gnpe_time_shifts']
        transforms.append(GNPEDetectorTimes(
            ifo_list, d['kernel_kwargs'], d['exact_equiv'],
            std=standardization_dict['std']['geocent_time']))
        gnpe_proxy_dim += transforms[-1].gnpe_proxy_dim
    transforms.append(ProjectOntoDetectors(ifo_list, domain, ref_time))
    transforms.append(SampleNoiseASD(asd_dataset))
    transforms.append(WhitenAndScaleStrain(domain.noise_std))
    transforms.append(AddWhiteNoiseComplex())
    transforms.append(SelectStandardizeRepackageParameters(standardization_dict))
    transforms.append(RepackageStrainsAndASDS(
        train_settings['transform_settings']['detectors']))
    if gnpe_proxy_dim == 0:
        selected_keys = ['parameters', 'waveform']
    else:
        selected_keys = ['parameters', 'waveform', 'gnpe_proxies']
    transforms.append(UnpackDict(selected_keys=selected_keys))

    # set wfd transforms to the composition of the above transforms
    wfd.transform = torchvision.transforms.Compose(transforms)

    return wfd


def build_train_and_test_loaders(train_settings, wfd):
    train_dataset, test_dataset = split_dataset_into_train_and_test(
        wfd, train_settings['train_settings']['train_fraction'])

    # build dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_settings['train_settings']['batch_size'],
        shuffle=True,
        pin_memory=True,
        num_workers=train_settings['train_settings']['num_workers'],
        worker_init_fn=lambda _: np.random.seed(
            int(torch.initial_seed()) % (2 ** 32 - 1)))
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_settings['train_settings']['batch_size'],
        shuffle=False,
        pin_memory=True,
        num_workers=train_settings['train_settings']['num_workers'],
        worker_init_fn=lambda _: np.random.seed(
            int(torch.initial_seed()) % (2 ** 32 - 1)))

    return train_loader, test_loader