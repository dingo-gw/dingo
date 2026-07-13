# Inference on an injection

A simple end-to-end test of a trained model is to create an injection consistent
with the training data and run inference on it. The
{py:class}`~dingo.gw.injection.Injection` is instantiated from the metadata of the
trained network (see [inference](inference.md)). An ASD dataset must also be
specified; here we take the fiducial dataset the network was trained on.

```python
from dingo.core.posterior_models import build_model_from_kwargs
import dingo.gw.injection as injection
from dingo.gw.noise.asd_dataset import ASDDataset

main_pm = build_model_from_kwargs(
    filename="/path/to/main_network.pt", device="cuda", load_training_info=False
)
init_pm = build_model_from_kwargs(
    filename="/path/to/init_network.pt", device="cuda", load_training_info=False
)

injection_generator = injection.Injection.from_posterior_model_metadata(main_pm.metadata)
asd_fname = main_pm.metadata["train_settings"]["training"]["stage_0"]["asd_dataset_path"]
detectors = main_pm.metadata["train_settings"]["data"]["detectors"]
injection_generator.asd = ASDDataset(file_name=asd_fname, ifos=detectors)

intrinsic_parameters = {
    "chirp_mass": 35,
    "mass_ratio": 0.5,
    "a_1": 0.3,
    "a_2": 0.5,
    "tilt_1": 0.0,
    "tilt_2": 0.0,
    "phi_jl": 0.0,
    "phi_12": 0.0,
}

extrinsic_parameters = {
    "phase": 0.0,
    "theta_jn": 2.3,
    "geocent_time": 0.0,
    "luminosity_distance": 400.0,
    "ra": 0.0,
    "dec": 0.0,
    "psi": 0.0,
}

theta = {**intrinsic_parameters, **extrinsic_parameters}
strain_data = injection_generator.injection(theta)
```

This example uses a [GNPE](gnpe.md) model pair, so the sampler is built from the
initialization and main networks; for a plain-NPE model, use
`GWComposedSampler.from_model(model, strain_data)` instead.

```python
from dingo.gw.inference.sampler import GWComposedSampler

sampler = GWComposedSampler.from_gnpe_models(
    init_pm, main_pm, strain_data, num_iterations=30
)
sampler.run_sampler(num_samples=50_000, batch_size=10_000)
result = sampler.to_result()
result.plot_corner()
```
