import h5py
import yaml
import argparse
import ast
from os.path import join
from dingo.core.dataset import recursive_hdf5_load, recursive_hdf5_save

parser = argparse.ArgumentParser(description='Train Dingo.')
parser.add_argument('--train_dir', required=True,
                    help='Train directory for Dingo training. Contains'
                         'train_settings.yaml file, used for logging.')

parser.add_argument('--new_dataset_path', required=True,
                    help='Path of the new ASD dataset file')

args = parser.parse_args()

with open(join(args.train_dir, 'train_settings.yaml'), 'r') as fp:
    train_settings = yaml.safe_load(fp)


asd_file = h5py.File(train_settings['asd_dataset_path'])
loaded_dict = recursive_hdf5_load(asd_file)

new_file = h5py.File(args.new_dataset_path, "w")

data_dict = {"asds": {}, "gps_times": {}}

data_dict['asds']["H1"] = loaded_dict["asds_H1"]
data_dict['asds']["L1"] = loaded_dict["asds_L1"]
data_dict["gps_times"] = loaded_dict["gps_times"]

domain_dict_old = ast.literal_eval(asd_file.attrs["metadata"])["domain_dict"]

domain_dict = {
    "type": "FrequencyDomain",
    "f_min": domain_dict_old["kwargs"]["f_min"],
    "f_max": domain_dict_old["kwargs"]["f_max"],
    "delta_f": domain_dict_old["kwargs"]["delta_f"]
}
new_file.attrs["settings"] = str({"domain_dict": domain_dict})

recursive_hdf5_save(new_file, data_dict)

