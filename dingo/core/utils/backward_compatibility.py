def update_model_config(model_settings: dict):
    """
    Update the model settings to ensure backwards compatibility with networks
    trained using previous versions of Dingo.

    Parameters
    ----------
    model_settings: dict
        Model settings to be updated.
    """
    if model_settings.get("type") == "nsf+embedding":
        model_settings["posterior_model_type"] = "normalizing_flow"
        del model_settings["type"]
        model_settings["posterior_kwargs"] = model_settings["nsf_kwargs"]
        del model_settings["nsf_kwargs"]
        model_settings["embedding_kwargs"] = model_settings["embedding_net_kwargs"]
        del model_settings["embedding_net_kwargs"]
