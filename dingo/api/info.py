# import yaml
# from dingo.gw.waveform_dataset import WaveformDataset
#
# def get_info(filepath, classname = WaveformDataset):
#     """
#     Prints metadata of object stored in filepath in yaml style.
#     TODO: Generalize to loaded objects
#     """
#     if classname == WaveformDataset:
#         wfd = WaveformDataset(filepath)
#         print(yaml.dump(wfd.data_settings))
#     else:
#         raise NotImplementedError