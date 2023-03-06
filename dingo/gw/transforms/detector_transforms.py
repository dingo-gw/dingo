import math
import numpy as np
import torch
import pandas as pd
from bilby.gw.detector.interferometer import Interferometer
from lal import GreenwichMeanSiderealTime
from typing import Union
from bilby.gw.detector import calibration
from bilby.gw.prior import CalibrationPriorDict


CC = 299792458.0
NUM_CALIBRATION_NODES = 10


def time_delay_from_geocenter(
    ifo: Interferometer,
    ra: Union[float, np.ndarray, torch.Tensor],
    dec: Union[float, np.ndarray, torch.Tensor],
    time: float,
):
    """
    Calculate time delay between ifo and geocenter. Identical to method
    ifo.time_delay_from_geocenter(ra, dec, time), but the present implementation allows
    for batched computation, i.e., it also accepts arrays and tensors for ra and dec.

    Implementation analogous to bilby-cython implementation
    https://git.ligo.org/colm.talbot/bilby-cython/-/blob/main/bilby_cython/geometry.pyx,
    which is in turn based on XLALArrivaTimeDiff in TimeDelay.c.

    Parameters
    ----------
    ifo: bilby.gw.detector.interferometer.Interferometer
        bilby interferometer object.
    ra: Union[float, np.array, torch.Tensor]
        Right ascension of the source in radians. Either float, or float array/tensor.
    dec: Union[float, np.array, torch.Tensor]
        Declination of the source in radians. Either float, or float array/tensor.
    time: float
        GPS time in the geocentric frame.

    Returns
    -------
    float: Time delay between the two detectors in the geocentric frame
    """
    # check that ra and dec are of same type and length
    if type(ra) != type(dec):
        raise ValueError(
            f"ra type ({type(ra)}) and dec type ({type(dec)}) don't match."
        )
    if isinstance(ra, (np.ndarray, torch.Tensor)):
        if len(ra.shape) != 1:
            raise ValueError(f"Only one axis expected for ra and dec, got multiple.")
        if ra.shape != dec.shape:
            raise ValueError(
                f"Shapes of ra ({ra.shape}) and dec ({dec.shape}) don't match."
            )

    if isinstance(ra, (float, np.float32, np.float64)):
        return ifo.time_delay_from_geocenter(ra, dec, time)

    elif isinstance(ra, (np.ndarray, torch.Tensor)) and len(ra) == 1:
        return ifo.time_delay_from_geocenter(ra[0], dec[0], time)

    else:
        if isinstance(ra, np.ndarray):
            sin = np.sin
            cos = np.cos
        elif isinstance(ra, torch.Tensor):
            sin = torch.sin
            cos = torch.cos
        else:
            raise NotImplementedError(
                "ra, dec must be either float, np.ndarray, or torch.Tensor."
            )

        gmst = math.fmod(GreenwichMeanSiderealTime(float(time)), 2 * np.pi)
        phi = ra - gmst
        theta = np.pi / 2 - dec
        sintheta = sin(theta)
        costheta = cos(theta)
        sinphi = sin(phi)
        cosphi = cos(phi)
        detector_1 = ifo.vertex
        detector_2 = np.zeros(3)
        return (
            (detector_2[0] - detector_1[0]) * sintheta * cosphi
            + (detector_2[1] - detector_1[1]) * sintheta * sinphi
            + (detector_2[2] - detector_1[2]) * costheta
        ) / CC


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
            dt = time_delay_from_geocenter(ifo, ra, dec, self.ref_time)
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


class ApplyCalibrationUncertainty(object):
    """
    Based off: 

    https://dcc.ligo.org/LIGO-T1400682/public

    Usually gravitational wave data is in the form 
    
    d(f) = h_obs(f) + n(f)                                          (1)

    Where d is the data, h is the waveform and n is the noise. However, since
    the detector is not perfectly calibrated, there are corrections to the
    waveform in the form
    
    h_obs(f) = h(f) * (1 + \delta A(f)) * exp(i \delta \phi(f))      (2)

    We can parameterize A(f) and \phi(f) with a cubic spline i.e.

    \delta A(f) = spline(f; {f_i, \delta A_i})                       (3)
    \delta \phi(f) = spline(f; {f_i, \delta \phi_i})                 (4)

    The \A_i and \phi_i are drawn from gaussians centered at 0 with standard
    deviations determined by the calibration envelope which varies event to
    event. This method draws multiple splines and multiplies the waveform 
    by the splines. Later when computing the likelihoods, we can marginalize
    over these waveforms thereby mitigating the effects of calibration 
    uncertainties in the detectors.  
    

    calibration_marginalization_kwargs: dict
        Calibration marginalization kwargs. If None no calibration marginalization is
        used. This should contain a dict with
        {"num_calibration_curves": 100, "calibration_lookup_table": {"H1": filepath, "L1"...}}.
        Optionally, you can also set "calibration_lookup_table" to None
    """

    def __init__(
        self, ifo_list, data_domain, calibration_envelope, num_calibration_curves
    ):
        """
        Initialize calibration marginalization. This requires the user to give
        an event specific calibration curve.

        Calibration marginalization takes a median and sigma (of amplitude and
        phase) for each point in frequency space. Then it creates a spline on
        the median and sigma. NOTE this is a DIFFERENT spline from the cubic
        spline that will be generated later. This spline is just needed to
        create priors on the the node points for the calibration cubic spline.
        This spline will give you the node points, i.e. f_i (log spaced) and
        Priors(\delta A_i, \delta \psi_i).

        These node points are the calibration parameters. From these node points
        and priors you can draw values for \delta A_i and \delta \psi_i which
        will give you a calibration curve. This is the calibration curve which
        you then multiply against the waveform.
        """

        self.ifo_list = ifo_list
        self.num_calibration_curves = num_calibration_curves

        self.data_domain = data_domain
        self.calibration_prior = {}
        if all([s.endswith(".txt") for s in calibration_envelope.values()]):
            # Generating .h5 lookup table from priors in .txt file
            self.calibration_envelope = calibration_envelope
            for i, ifo in enumerate(self.ifo_list):
                # Setting calibration model to cubic spline
                ifo.calibration_model = calibration.CubicSpline(
                    f"recalib_{ifo.name}_",
                    minimum_frequency=data_domain.f_min,
                    maximum_frequency=data_domain.f_max,
                    n_points=NUM_CALIBRATION_NODES,
                )

                # Setting priors
                # What this will do is take the the calibration envelope and set
                # a spline on the median and sigma of the amplitude and phase.
                # Then in log frequency it will setup node points say at
                # frequency points, $f_i$.  Then for each node point f_i, it
                # will create a gaussian prior according to the spline of the
                # median and sigma found earlier
                self.calibration_prior[
                    ifo.name
                ] = CalibrationPriorDict.from_envelope_file(
                    self.calibration_envelope[ifo.name],
                    self.data_domain.f_min,
                    self.data_domain.f_max,
                    NUM_CALIBRATION_NODES,
                    ifo.name,
                )

        else:
            raise Exception("Calibration envelope must be specified in a .txt file!")

    def __call__(self, input_sample):
        sample = input_sample.copy()
        for ifo in self.ifo_list:
            calibration_parameter_draws, calibration_draws = {}, {}
            # Sampling from prior
            calibration_parameter_draws[ifo.name] = pd.DataFrame(
                self.calibration_prior[ifo.name].sample(self.num_calibration_curves)
            )
            calibration_draws[ifo.name] = np.zeros(
                (
                    self.num_calibration_curves,
                    len(self.data_domain.sample_frequencies),
                ),
                dtype=complex,
            )

            for i in range(self.num_calibration_curves):
                calibration_draws[ifo.name][
                    i, self.data_domain.frequency_mask
                ] = ifo.calibration_model.get_calibration_factor(
                    self.data_domain.sample_frequencies[
                        self.data_domain.frequency_mask
                    ],
                    prefix="recalib_{}_".format(ifo.name),
                    **calibration_parameter_draws[ifo.name].iloc[i],
                )

            # Multiplying the sample waveform in the interferometer according to
            # the calibration curve.  This is done by following the perscription
            # here:
            #
            # https://dcc.ligo.org/LIGO-T1400682 Eq 3 and 4
            #
            # We take the waveform h(f) and multiply it by C = (1 + \delta A(f))
            # \exp(i \delta \psi) i.e. h_obs(f) = C * h(f)
            # Here C is "calibration_draws"

            # Padding 0's to everything in the calibration array which is below f_min

            sample["waveform"][ifo.name] = (
                sample["waveform"][ifo.name] * calibration_draws[ifo.name]
            )

        return sample
