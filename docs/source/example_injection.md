# Inference on an injection 

A simple example is creating an injection consistent with what the
network was trained on, and then running Dingo on it. First one can instantiate
the {py:class}`dingo.gw.injection.Injection` using the metadata from the
{py:class}`dingo.core.models.posterior_model.PosteriorModel` (the trained network). An ASD dataset also needs to be specified,
one can take the fiducial asd dataset the network was trained on. 

```
from dingo.core.models import PosteriorModel
import dingo.gw.injection as injection
from dingo.gw.ASD_dataset.noise_dataset import ASDDataset

main_pm = PosteriorModel(
    device="cuda",
    model_filename="/path/to/main_network", 
    load_training_info=False
)

init_pm = PosteriorModel(
    device='cuda',
    model_filename="/path/to/init_network",
    load_training_info=False
)

injection_generator = injection.Injection.from_posterior_model_metadata(main_pm.metadata)
asd_fname = main_pm.metadata["train_settings"]["training"]["stage_0"]["asd_dataset_path"]
asd_dataset = ASDDataset(file_name=asd_fname)
injection_generator.asd = {k:v[0] for k,v in asd_dataset.asds.items()}

intrinsic_parameters = {
    "chirp_mass": 35,
    "mass_ratio": 0.5,
    "a_1": .3,
    "a_2": .5,
    "tilt_1": 0.,
    "tilt_2": 0.,
    "phi_jl": 0.,
    "phi_12": 0.
}

extrinsic_parameters = {
    'phase': 0.,
    'theta_jn': 2.3,
    'geocent_time': 0.,
    'luminosity_distance': 400.,
    'ra': 0.,
    'dec': 0.,
    'psi': 0.,
}

theta = {**intrinsic_parameters, **extrinsic_parameters}
strain_data = injection_generator.injection(theta)
```

Then one can create a injections and do inference on them.

```
from dingo.gw.inference.gw_samplers import GWSamplerGNPE, GWSampler

init_sampler = GWSampler(model=init_pm)
sampler = GWSamplerGNPE(model=main_pm, init_sampler=init_sampler, num_iterations=30)
sampler.context = strain_data
sampler.run_sampler(num_samples=50_000, batch_size=10_000)
result = sampler.to_result()
result.plot_corner()
```
