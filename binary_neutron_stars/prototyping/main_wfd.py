import yaml
import matplotlib.pyplot as plt
from dingo.gw.dataset import generate_dataset
from dingo.gw.transforms import HeterodynePhase

if __name__ == "__main__":
    num_processes = 10
    with open("waveform_dataset_settings.yaml", "r") as f:
        settings = yaml.safe_load(f)
    wfd = generate_dataset(settings, num_processes)
    wfd.to_file("./waveform_dataset_bns.hdf5")

    heterodyne = HeterodynePhase(wfd.domain)
    data = wfd[0]
    data_het = heterodyne(wfd[0])
    plt.plot(wfd.domain(), data["waveform"]["h_plus"])
    plt.plot(wfd.domain(), data_het["waveform"]["h_plus"])
    plt.show()
    plt.plot(data["waveform"]["h_plus"])
    plt.plot(data_het["waveform"]["h_plus"])
    plt.show()