import numpy as np
import os
import torch

from bilby.gw.detector import calibration


class GetDetectorTimes(object):
    """
    Compute the time shifts in the individual detectors based on the sky
    position (ra, dec), the geocent_time and the ref_time.
    """

    def __init__(self, ifo_list, ref_time):
        self.ifo_list = ifo_list
        self.ref_time = ref_time

    def __call__(self, input_sample):
        sample = input_sample.copy()
        # the line below is required as sample is a shallow copy of
        # input_sample, and we don't want to modify input_sample
        extrinsic_parameters = sample["extrinsic_parameters"].copy()
        ra = extrinsic_parameters["ra"]
        dec = extrinsic_parameters["dec"]
        geocent_time = extrinsic_parameters["geocent_time"]
        for ifo in self.ifo_list:
            if type(ra) == torch.Tensor:
                # computation does not work on gpu, so do it on cpu
                ra = ra.cpu()
                dec = dec.cpu()
            dt = ifo.time_delay_from_geocenter(ra, dec, self.ref_time)
            if type(dt) == torch.Tensor:
                dt = dt.to(geocent_time.device)
            ifo_time = geocent_time + dt
            extrinsic_parameters[f"{ifo.name}_time"] = ifo_time
        sample["extrinsic_parameters"] = extrinsic_parameters
        return sample


class ProjectOntoDetectors(object):
    """
    Project the GW polarizations onto the detectors in ifo_list. This does
    not sample any new parameters, but relies on the parameters provided in
    sample['extrinsic_parameters']. Specifically, this transform applies the
    following operations:

    (1) Rescale polarizations to account for sampled luminosity distance
    (2) Project polarizations onto the antenna patterns using the ref_time and
        the extrinsic parameters (ra, dec, psi)
    (3) Time shift the strains in the individual detectors according to the
        times <ifo.name>_time provided in the extrinsic parameters.
    """

    def __init__(self, ifo_list, domain, ref_time):
        self.ifo_list = ifo_list
        self.domain = domain
        self.ref_time = ref_time

    def __call__(self, input_sample):
        sample = input_sample.copy()
        # the line below is required as sample is a shallow copy of
        # input_sample, and we don't want to modify input_sample
        parameters = sample["parameters"].copy()
        extrinsic_parameters = sample["extrinsic_parameters"].copy()
        try:
            d_ref = parameters["luminosity_distance"]
            d_new = extrinsic_parameters.pop("luminosity_distance")
            ra = extrinsic_parameters.pop("ra")
            dec = extrinsic_parameters.pop("dec")
            psi = extrinsic_parameters.pop("psi")
            tc_ref = parameters["geocent_time"]
            assert tc_ref == 0, (
                "This should always be 0. If for some reason "
                "you want to save time shifted polarizations,"
                " then remove this assert statement."
            )
            tc_new = extrinsic_parameters.pop("geocent_time")
        except:
            raise ValueError("Missing parameters.")

        # (1) rescale polarizations and set distance parameter to sampled value
        hc = sample["waveform"]["h_cross"] * d_ref / d_new
        hp = sample["waveform"]["h_plus"] * d_ref / d_new
        parameters["luminosity_distance"] = d_new

        strains = {}
        for ifo in self.ifo_list:
            # (2) project strains onto the different detectors
            fp = ifo.antenna_response(ra, dec, self.ref_time, psi, mode="plus")
            fc = ifo.antenna_response(ra, dec, self.ref_time, psi, mode="cross")
            strain = fp * hp + fc * hc

            # (3) time shift the strain. If polarizations are timeshifted by
            #     tc_ref != 0, undo this here by subtracting it from dt.
            dt = extrinsic_parameters[f"{ifo.name}_time"] - tc_ref
            strains[ifo.name] = self.domain.time_translate_data(strain, dt)

        # Add extrinsic parameters corresponding to the transformations
        # applied in the loop above to parameters. These have all been popped off of
        # extrinsic_parameters, so they only live one place.
        parameters["ra"] = ra
        parameters["dec"] = dec
        parameters["psi"] = psi
        parameters["geocent_time"] = tc_new
        for ifo in self.ifo_list:
            param_name = f"{ifo.name}_time"
            parameters[param_name] = extrinsic_parameters.pop(param_name)

        sample["waveform"] = strains
        sample["parameters"] = parameters
        sample["extrinsic_parameters"] = extrinsic_parameters

        return sample


class TimeShiftStrain(object):
    """
    Time shift the strains in the individual detectors according to the
    times <ifo.name>_time provided in the extrinsic parameters.
    """

    def __init__(self, ifo_list, domain):
        self.ifo_list = ifo_list
        self.domain = domain

    def __call__(self, input_sample):
        sample = input_sample.copy()
        extrinsic_parameters = input_sample["extrinsic_parameters"].copy()

        strains = {}

        if isinstance(input_sample["waveform"], dict):
            for ifo in self.ifo_list:
                # time shift the strain
                strain = input_sample["waveform"][ifo.name]
                dt = extrinsic_parameters.pop(f"{ifo.name}_time")
                strains[ifo.name] = self.domain.time_translate_data(strain, dt)

        elif isinstance(input_sample["waveform"], torch.Tensor):
            strains = input_sample["waveform"]
            dt = [extrinsic_parameters.pop(f"{ifo.name}_time") for ifo in self.ifo_list]
            dt = torch.stack(dt, 1)
            strains = self.domain.time_translate_data(strains, dt)

        else:
            raise NotImplementedError(
                f"Unexpected type {type(input_sample['waveform'])}, expected dict or "
                f"torch.Tensor"
            )

        sample["waveform"] = strains
        sample["extrinsic_parameters"] = extrinsic_parameters

        return sample


class MultiplyCalibrationUncertainty(object):
    """
    calibration_marginalization_kwargs: dict
        Calibration marginalization kwargs. If None no calibration marginalization is
        used. This should contain a dict with
        {"num_calibration_curves": 100, "calibration_lookup_table": {"H1": filepath, "L1"...}}.
        Optionally, you can also set "calibration_lookup_table" to None
    """

    def __init__(self, ifo_list, data_domain, calibration_lookup_table):
        """
        Initialize calibration marginalization. This store the calibration curve prior will later be applied to
        the waveform. We can either specify what the calibration values are via a lookup table or randomly generate
        the fake curves based on a prior. The former is useful for when you have an event you are interested in.
        """

        self.ifo_list = ifo_list
        self.calibration_lookup_table = calibration_lookup_table
        self.data_domain = data_domain
        if self.calibration_lookup_table == "generate":
            raise NotImplementedError(
                "Random Generation of calibration curves not implemented yet"
            )
        else:
            self.calibration_lookup_table = calibration_lookup_table

    def __call__(self, input_sample):
        sample = input_sample.copy()
        sample["calibration_draw"] = {ifo.name:None for ifo in self.ifo_list}
        for ifo in self.ifo_list:
            if os.path.exists(self.calibration_lookup_table[ifo.name]):
                calibration_draw = calibration.read_calibration_file(
                    self.calibration_lookup_table[ifo.name],
                    self.data_domain.sample_frequencies,
                    number_of_response_curves=1,  # NOTE for now we are just pulling 1 calibration curve per posterior point
                    # In the future it may be beneficial to marginalize over this in a different way
                    starting_index=0,
                ).flatten() # Since we only have 1 response curve
            else:
                raise Exception(
                    f"Could not find calibration file '{self.calibration_lookup_table[ifo.name]}'"
                )


            # Multiplying the sample waveform in the interferometer according to the calibration curve
            # This is done by following the perscription here:
            #
            # https://dcc.ligo.org/LIGO-T1400682 Eq 3 and 4
            # 
            # We take the waveform h(f) and multiply it by C = (1 + \delta A(f)) \exp(i \delta \psi) 
            # i.e. h_obs(f) = C * h(f)
            # Here C is "calibration_draws"

            sample["waveform"][ifo.name] *= calibration_draw
            sample["calibration_draw"][ifo.name] = calibration_draw

        return sample
