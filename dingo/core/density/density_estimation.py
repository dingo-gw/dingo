import torch


class SampleDataset(torch.utils.data.Dataset):
    """
    Dataset class for unconditional density estimation.
    This is required, since the training method of dingo.core.models.PosteriorModel
    expects a tuple of (theta, *context) as output of the DataLoader, but here we have
    no context, so len(context) = 0. This SampleDataset therefore returns a tuple
    (theta, ) instead of just theta.
    """

    def __init__(self, data):
        self.data = data
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """Return the data and labels at the given index as a tuple of length 1."""
        return (self.data[index],)


if __name__ == "__main__":
    import yaml
    from os.path import dirname, join
    from torch.utils.data import DataLoader
    from dingo.core.utils.trainutils import RuntimeLimits
    import numpy as np

    # load settings
    f = "/Users/maxdax/Documents/Projects/GW-Inference/dingo/datasets/dingo_samples/01_Pv2/density_settings.yaml"
    with open(f, "r") as fp:
        settings = yaml.safe_load(fp)

    # load samples from dingo output
    import pandas as pd

    samples = pd.read_pickle(settings["data"]["sample_file"])
    if "parameters" in settings["data"] and settings["data"]["parameters"]:
        parameters = settings["data"]["parameters"]
    else:
        parameters = list(samples.keys())
    samples = np.array(samples[parameters])
    num_samples, num_params = samples.shape
    mean, std = np.mean(samples, axis=0), np.std(samples, axis=0)
    # normalized torch samples
    samples_torch = torch.from_numpy((samples - mean) / std).float()

    # set up density estimation network
    from dingo.core.models import PosteriorModel

    settings["model"]["input_dim"] = num_params
    settings["model"]["context_dim"] = None
    model = PosteriorModel(metadata={"train_settings": settings}, device="cpu")
    model.optimizer_kwargs = settings["training"]["optimizer"]
    model.scheduler_kwargs = settings["training"]["scheduler"]
    model.initialize_optimizer_and_scheduler()

    # set up dataloaders
    num_train_samples = int(num_samples * settings["training"]["train_fraction"])
    train_loader = DataLoader(
        SampleDataset(samples_torch[:num_train_samples]),
        batch_size=settings["training"]["batch_size"],
        shuffle=True,
        num_workers=settings["training"]["num_workers"],
    )
    test_loader = DataLoader(
        SampleDataset(samples_torch[num_train_samples:]),
        batch_size=settings["training"]["batch_size"],
        shuffle=True,
        num_workers=settings["training"]["num_workers"],
    )

    # train model
    runtime_limits = RuntimeLimits(
        epoch_start=0,
        max_epochs_total=settings["training"]["epochs"],
    )
    model.train(
        train_loader,
        test_loader,
        train_dir=dirname(f),
        runtime_limits=runtime_limits,
    )

    model.model.eval()
    with torch.no_grad():
        new_samples = model.model.sample(num_samples=10000)
    new_samples = new_samples.cpu().numpy() * std + mean
    new_samples = pd.DataFrame(new_samples, columns=parameters)
    test_samples = pd.DataFrame(samples[num_train_samples:], columns=parameters)

    from dingo.gw.inference.visualization import generate_cornerplot
    generate_cornerplot(
        {"samples": test_samples, "color": "blue", "name": "test samples"},
        {"samples": new_samples, "color": "orange", "name": "new samples"},
        filename=join(dirname(f), "out.pdf")
    )

    print("done")
