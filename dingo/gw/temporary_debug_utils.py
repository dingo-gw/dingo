import numpy as np

def save_training_injection(outname, pm, data, idx=0):
    """
    For debugging: extract a training injection. To be used inside train or test loop.
    """
    param_names = pm.metadata["train_settings"]["data"]["inference_parameters"]
    mean = pm.metadata["train_settings"]["data"]["standardization"]["mean"]
    std = pm.metadata["train_settings"]["data"]["standardization"]["std"]
    params = {p: data[0][idx, idx_p] for idx_p, p in enumerate(param_names)}
    params = {p: float(v * std[p] + mean[p]) for p, v in params.items()}

    from dingo.gw.domains import build_domain_from_model_metadata

    domain = build_domain_from_model_metadata(pm.metadata)
    detectors = pm.metadata["train_settings"]["data"]["detectors"]
    d = np.array(data[1])
    asds = {
        ifo: 1 / d[idx, idx_ifo, 2] * 1e-23 for idx_ifo, ifo in enumerate(detectors)
    }
    strains = {
        ifo: (d[idx, idx_ifo, 0] + 1j * d[idx, idx_ifo, 1])
        * (asds[ifo] * domain.noise_std)
        for idx_ifo, ifo in enumerate(detectors)
    }

    out_data = {"parameters": params, "asds": asds, "strains": strains}
    np.save(outname, out_data)

    from dingo.gw.injection import GWSignal

    signal = GWSignal(
        pm.metadata["dataset_settings"]["waveform_generator"],
        domain,
        domain,
        pm.metadata["train_settings"]["data"]["detectors"],
        pm.metadata["train_settings"]["data"]["ref_time"],
    )
    params_2 = params.copy()
    params_2["phase"] = (params_2["phase"] + np.pi/2.) % (2 * np.pi)
    params_3 = {p: v * 0.99 for p, v in params.items()}
    sample = signal.signal(params)
    sample_2 = signal.signal(params_2)
    sample_3 = signal.signal(params_3)

    import matplotlib.pyplot as plt

    plt.plot(np.abs(sample["waveform"]["H1"])[domain.min_idx:])
    plt.plot(np.abs(strains["H1"]), lw=0.8)
    plt.show()

    plt.plot(sample["waveform"]["H1"][domain.min_idx:])
    plt.plot(strains["H1"], lw=0.8)
    # plt.plot(sample_2["waveform"]["H1"][domain.min_idx:])
    plt.show()