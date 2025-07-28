import torchvision.transforms

from dingo.core.transforms import DictToArray, PadMask, DictToEventArray
from dingo.gw.transforms import StandardizeParameters, UnpackDict, ToTorch


def set_train_transforms(population_model, data_settings, settings_pm_single_event):
    print(f"Setting train transforms.")
    transforms = []

    # If the standardization factors have already been set, use those. Otherwise,
    # calculate them, and save them within the data settings.
    try:
        # standardization_dict = data_settings["standardization"]
        standardization_dict = population_model["train_settings"]["data"]["standardization"] # TODO: check with standard convention
        print("Using previously-calculated parameter standardizations.")
    except KeyError:
        print("Calculating new parameter standardizations.")
        standardization_dict = population_model.hyperparameter_mean_std()
        data_settings["standardization"] = standardization_dict

    transforms.append(
        StandardizeParameters(
            mu=standardization_dict["mean"],
            std=standardization_dict["std"],
            key="hyperparameters",
        )
    )

    standardization_dict_parameters = {}
    
    for x in ['mean', 'std']:
        standardization_dict_parameters[x] = \
              {k:v for k,v in settings_pm_single_event['train_settings']['data']['standardization'][x].items() if k in population_model.inference_parameters}

        if 'matched_filter_snr' in standardization_dict_parameters[x]:
            # Remove matched_filter_snr from standardization, since it is not used in the population model.
            del standardization_dict_parameters[x]['matched_filter_snr']
            
    data_settings["standardization_single_events"] = standardization_dict_parameters
    transforms.append(
        StandardizeParameters(
            mu=standardization_dict_parameters["mean"],
            std=standardization_dict_parameters["std"],
            key="parameters",
        )
    )

    # The convention is that the ordering of parameters is the same as that of the
    # standardization_dict. It may be preferable to introduce a list of inference
    # parameters, as we do in dingo.gw.
    transforms.append(DictToArray("hyperparameters"))
    transforms.append(DictToEventArray("parameters"))
    transforms.append(ToTorch(device="cpu"))
    transforms.append(UnpackDict(["hyperparameters", "parameters", "size"]))
    
    # The transformer requires sequences to be padded up to the maximum length. This
    # does so, and produces a corresponding mask (True = mask out token).
    # Commented this out, since the embedding and mask generation is done now in training loop. 
    # transforms.append(PadMask(1, 0, population_model.maximum_population_size))

    population_model.transform = torchvision.transforms.Compose(transforms)
