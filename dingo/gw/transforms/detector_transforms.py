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
            dt = ifo.time_delay_from_geocenter(ra, dec, self.ref_time)
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

        # Add extrinsic extrinsic parameters corresponding to the transformations
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
