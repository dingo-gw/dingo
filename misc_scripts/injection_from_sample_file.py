import numpy as np
from types import SimpleNamespace
from os.path import join, dirname
import argparse

from dingo.core.dataset import DingoDataset
from dingo.core.samples_dataset import SamplesDataset
from dingo.gw.inference.injection import Injection

parser = argparse.ArgumentParser(
    description="Generate injection based off of a dingo samples file.",
)
parser.add_argument(
    "--samples_file",
    type=str,
    required=True,
    help="Path to dingo samples file. Used to extract asds and metadata.",
)
parser.add_argument(
    "--injection_file",
    type=str,
    required=True,
    help="Name of out file. Saved in same directory as args.samples_file.",
)
args = parser.parse_args()

sd = SamplesDataset(args.samples_file)
pm = SimpleNamespace(metadata=sd.settings)
injection_gen = Injection.from_posterior_model(pm)
injection_gen.asd = sd.context["asds"]

theta_injection = {
    k: v
    for k, v in sd.samples.iloc[0].to_dict().items()
    if k in injection_gen.prior.keys()
}
injection_data = injection_gen.injection(theta_injection)
injection_data.pop("extrinsic_parameters")

# sanity check
for ifo in sd.context["waveform"]:
    wf_white_0 = sd.context["waveform"][ifo] / sd.context["asds"][ifo]
    wf_white_1 = injection_data["waveform"][ifo] / injection_data["asds"][ifo]
    assert abs(1 - np.std(wf_white_0.real) / np.std(wf_white_1.real)) < 0.05
    assert abs(1 - np.std(wf_white_0.imag) / np.std(wf_white_1.imag)) < 0.05
    assert np.all(sd.context["asds"][ifo] == injection_data["asds"][ifo])
assert theta_injection == {
    k: injection_data["parameters"][k] for k in theta_injection.keys()
}

injection_dataset = DingoDataset(
    dictionary={
        "event_data": injection_data,
        "event_metadata": {**sd.settings["event"], "time_event": None},
        "settings": {"injection_pm_metadata": pm.metadata},
    },
    data_keys=["event_data", "event_metadata"],
)
injection_dataset.to_file(join(dirname(args.samples_file), args.injection_file))
# test = DingoDataset(
#     join(dirname(args.samples_file), args.injection_file),
#     data_keys=["event_data", "event_settings"],
# )
