"""Util functions for running bayestar based on a dingo results file."""

import numpy as np
import pycbc
from pycbc.types.frequencyseries import FrequencySeries
import lal
import ligo.skymap
import ligo.skymap.bayestar
import ligo.skymap.bayestar.interpolation
import ligo.skymap.io.events
from typing import Optional, Dict, List
from bilby.gw.conversion import generate_mass_parameters
from bilby.gw.detector import InterferometerList

from dingo.gw.domains import FrequencyDomain
from dingo.gw.result import Result
from dingo.gw.transforms.detector_transforms import time_delay_from_geocenter
from dingo.gw.waveform_generator import WaveformGenerator


def index_cyclic(array, lower, upper):
    """Numpy indexing array[lower:upper], but cyclic."""
    if len(array.shape) != 1:
        raise NotImplementedError("Only implemented for 1d arrays.")
    n = len(array)
    lower = lower % n
    upper = upper % n
    if lower > upper:
        part1 = array[lower:]
        part2 = array[:upper]
        return np.concatenate([part1, part2])
    else:
        return array[lower:upper]


def crop_snr_series(
    snr_series: pycbc.types.timeseries.TimeSeries,
    duration: float = 0.1,
    time_lower: Optional[float] = None,
    time_upper: Optional[float] = None,
):
    """Crop the complex snr series to specified duration, centered around its maximum.

    Parameters
    ----------
    snr_series: pycbc.types.timeseries.TimeSeries
        complex snr time series
    duration: float = 0.1
        duration (in seconds) of the cropped snr time series
    time_lower: Optional[float] = None
        optional lower bound for maximum
    time_upper: Optional[float] = None
        optional upper bound for maximum

    Returns
    -------
    snr_series_cropped: np.ndarray
        snr time series, cropped to duration
    delta_t: float
        delta_t of returned time series
    relative_time_offset: float
        offset of the cropped snr_series, compared to the input one
    """
    # find max bin of snr_series
    snr_series_abs = np.abs(snr_series.data)
    if time_lower is not None:
        snr_series_abs[np.where(snr_series.sample_times < time_lower)[0]] = 0
    if time_upper is not None:
        snr_series_abs[np.where(snr_series.sample_times > time_upper)[0]] = 0
    if np.max(snr_series_abs) == 0:
        raise ValueError
    idx_max = np.argmax(snr_series_abs)

    # number of bins
    n = int(duration / snr_series.delta_t)
    if n % 2 == 0:
        n += 1

    # crop window of length n, centered around idx_max
    lower = idx_max - n // 2
    upper = idx_max + n // 2 + 1
    snr_series_cropped = index_cyclic(snr_series.data, lower, upper)

    # check that cropped series has correct length and that max is in center
    assert len(snr_series_cropped) == n
    if time_lower is None and time_upper is None:
        assert np.argmax(np.abs(snr_series_cropped)) == n // 2, np.argmax(
            np.abs(snr_series_cropped)
        )

    # get time offset of cropped snr series
    delta_t = snr_series.delta_t
    relative_time_offset = lower * delta_t

    return snr_series_cropped, delta_t, relative_time_offset


class DingoSingleEvent(ligo.skymap.io.events.SingleEvent):
    """Single event class wrapper for interfacing with Bayestar.

    A single event corresponds to a single detector.
    """

    def __init__(
        self,
        ifo: str,
        gps_time_data: float,
        data: np.ndarray,
        asd: np.ndarray,
        template: np.ndarray,
        domain: FrequencyDomain,
        duration: float = 0.1,
        f_lower: Optional[float] = None,
        f_upper: Optional[float] = None,
        t_center: Optional[float] = None,
        t_width: Optional[float] = None,
    ):
        """Initializes DingoSingleEvent, computing among others the snr time series.

        Parameters
        ----------
        ifo: str
            interferometer name, e.g., "H1"
        gps_time_data: float
            gps time of the data array
        data: np.ndarray
            complex frequency domain strain data, must be compatible with domain
        asd: np.ndarray
            asd in frequency domain, must be compatible with domain
        template: np.ndarray
            waveform template in frequency domain
        domain: FrequencyDomain
            frequency domain of data, asd and template
        duration: float
            duration of the snr time series
        f_lower: float
            lower frequency bound for computation of snr series
        f_upper: float
            upper frequency bound for computation of snr series
        t_center: float
            gps time, center of the search for the peak of the snr series,
            only used if t_width is set
        t_width: float
            width of the window (centered around t_center) for the search of the
        # t_lower: float
        #     lower gps time bound for peak in snr series
        # t_upper: float
        #     upper gps time bound for peak in snr series
        """
        if not (len(domain) == len(data) == len(asd) == len(template)):
            raise ValueError(
                f"Domain, data, asd and template should all have the same lengths, "
                f"got {len(domain)}, {len(data)}, {len(asd)}, {len(template)}."
            )

        # store input args
        self._detector = ifo
        self._gps_time_data = gps_time_data
        self._data = data
        self._asd = np.copy(asd)
        self._asd[: domain.min_idx] = 1
        self._template = template
        self._domain = domain
        self._duration = duration
        self._f_lower = f_lower
        self._f_upper = f_upper
        if t_width is not None:
            self._t_lower = t_center - t_width / 2
            self._t_upper = t_center + t_width / 2
            self._time_shift_data()
        else:
            self._t_lower = None
            self._t_upper = None

        # these attributes are added below by compute_snr_series
        self._snr_series = None
        self._snr_series_cropped = None
        self._gps_time_merger = None
        self._snr = None
        self._phase = None
        self._snr_series_lal = None
        self._psd_lal = None

        # compute snr series and store the corresponding attributes
        self.compute_snr_series()

    def _time_shift_data(self):
        assert self._t_lower is not None and self._t_upper is not None
        if self._t_lower < self._gps_time_data:
            t_shift = self._gps_time_data - self._t_lower
            self._gps_time_data -= t_shift
            self._data = self._domain.time_translate_data(self._data, t_shift)

    def compute_snr_series(self, print_info: bool = False):
        """Compute the snr time series based on the strain data and template attributes.

        This generates the following attributes.
            self._snr_series: full snr time series, as pycbc FrequencySeries
            self._snr_series_cropped: snr series cropped to duration, centered around
                maximum snr
            self._epoch: gps start time of self._snr_series_cropped
            self._gps_time_merger: gps time of the merger
            self._snr: maximum snr (interpolated)
            self._phase: phase at maximum snr (interpolated)
            self._snr_series_lal: lal object of cropped snr series
            self._psd_lal: lal object of psd

        Parameters
        ----------
        print_info:

        """
        # compute snr series with template
        template = FrequencySeries(self._template, delta_f=self._domain.delta_f)
        data = FrequencySeries(self._data, delta_f=self._domain.delta_f)
        self._snr_series = pycbc.filter.matchedfilter.matched_filter(
            template=template / self._asd,  # whitened template
            data=data / self._asd,  # whitened data
            low_frequency_cutoff=self._f_lower,
            high_frequency_cutoff=self._f_upper,
        )
        if np.isnan(self._snr_series.data).any():
            raise ValueError("Got nan values in snr series, is f_lower too small?")

        # crop snr series to duration, centered around maximum
        time_lower = (
            (self._t_lower - self._gps_time_data) if self._t_lower is not None else None
        )
        time_upper = (
            (self._t_upper - self._gps_time_data) if self._t_upper is not None else None
        )
        self._snr_series_cropped, delta_t, relative_time_offset = crop_snr_series(
            self._snr_series,
            self._duration,
            time_lower=time_lower,
            time_upper=time_upper,
        )

        # absolute start time of the SNR time series
        self._epoch = self._gps_time_data + relative_time_offset

        # check time shifts
        t_max = self._snr_series.delta_t * np.argmax(np.abs(self._snr_series.data))
        t_max_cropped = relative_time_offset + delta_t * np.argmax(
            np.abs(self._snr_series_cropped)
        )
        if self._t_lower is None and self._t_upper is None:
            assert np.allclose(t_max, t_max_cropped), t_max - t_max_cropped

        # interpolate maximum for time, snr and phase at interpolated maximum
        n = len(self._snr_series_cropped)
        i_interp, z_interp = ligo.skymap.bayestar.interpolation.interpolate_max(
            n // 2, self._snr_series_cropped, n // 2, method="lanczos"
        )
        self._gps_time_merger = self._epoch + delta_t * i_interp
        self._snr = np.abs(z_interp)
        self._phase = np.angle(z_interp)

        # Convert snr series and psd to lal
        self._snr_series_lal = lal.CreateCOMPLEX8TimeSeries(
            "snr", self._epoch, 0, delta_t, lal.DimensionlessUnit, n
        )
        self._snr_series_lal.data.data = self._snr_series_cropped
        self._psd_lal = lal.CreateREAL8FrequencySeries(
            "psd", 0, 0, self._domain.delta_f, lal.SecondUnit, len(self._asd)
        )
        self._psd_lal.data.data = self._asd ** 2

        if print_info:
            print("Discrete / Interpolated:")
            print(f"Argmax index: {n // 2} / {i_interp:.3f}")
            print(
                f"Argmax time (= merger time): "
                f"{self._gps_time_data + t_max_cropped:.3} / {self._time:.3}"
            )
            print(
                f"SNR: {np.abs(self._snr_series_cropped[n // 2]):.3f} "
                f"/ {np.abs(z_interp):.3f}"
            )
            print(
                f"Phase: {np.angle(self._snr_series_cropped[n // 2]):.3f} "
                f"/ {np.angle(z_interp):.3f}"
            )

    @property
    def detector(self):
        return self._detector

    @property
    def snr(self):
        return self._snr

    @property
    def phase(self):
        return self._phase

    @property
    def time(self):
        return self._gps_time_merger

    @property
    def zerolag_time(self):
        return self._gps_time_merger

    @property
    def psd(self):
        return self._psd_lal

    @property
    def snr_series(self):
        return self._snr_series_lal


class DingoEvent(ligo.skymap.io.events.Event):
    """Event class wrapper for interfacing with Bayestar."""

    def __init__(
        self, singles: List[DingoSingleEvent], template_args: Dict[str, float]
    ):
        """Initializes DingoEvent.

        Parameters
        ----------
        singles: Tuple with DingoSingleEvent instances, one per detector
        template_args: Dict with parameters of template, containing mass1 and mass2
        """
        self._singles = singles
        self._template_args = template_args

    @property
    def singles(self):
        return self._singles

    @property
    def template_args(self):
        return self._template_args

    @classmethod
    def from_dingo_result(
        cls,
        dingo_result: Result,
        max_likelihood_template: bool = True,
        ifos: Optional[List[str]] = None,
        duration: float = 0.1,
        f_lower: Optional[float] = None,
        f_upper: Optional[float] = None,
        t_search_window_width: Optional[float] = None,
    ):
        """Instantiate a DingoEvent based on a dingo result instance.

        This uses the strain and asd data from dingo_result.context and a template
        generated from inferred dingo parameters.

        Parameters
        ----------
        dingo_result: Result
            dingo result file
        max_likelihood_template: bool = True
            if true, use maximum likelihood parameters for template, otherwise sample
            random parameters
        ifos: Optional[List[str]] = None
            interferometers used, if None use interferometers from dingo result
        single_event_kwargs:
            kwargs passed to DingoSingleEvent: duration, f_lower, f_upper, time_window

        """
        # get parameters theta for template
        if max_likelihood_template:
            # use template with maximum likelihood
            try:
                idx = np.argmax(dingo_result.samples["log_likelihood"])
                theta = dingo_result.samples.iloc[idx].to_dict()
                theta = generate_mass_parameters(theta)
                print(f"Using maximum likelihood parameters for template: {theta}")
            except KeyError:
                raise ValueError(
                    "Can only use max_likelihood_template if likelihoods are available, "
                    "but samples in dingo results don't have log_likelihood column."
                )
        else:
            # Draw random paramter sample. If dingo_samples have a "weights" column,
            # sample from the weighted distribution.
            weights = dingo_result.samples.get("weights", None)
            theta = dingo_result.samples.sample(1, weights=weights).iloc[0].to_dict()
            theta = generate_mass_parameters(theta)
            print(f"Using sampled parameters: {theta}")

        # generate template
        domain = getattr(dingo_result.domain, "base_domain", dingo_result.domain)
        wfg = WaveformGenerator(
            domain=domain,
            **dingo_result.metadata["dataset_settings"]["waveform_generator"],
        )
        template = wfg.generate_hplus_hcross({**theta, **dict(phase=0)})["h_plus"]

        if f_lower is None:
            f_lower = domain.f_min
        if f_upper is None:
            f_upper = domain.f_upper

        # set up interferometer list
        if ifos is None:
            ifos = dingo_result.context["asds"].keys()
        ifo_list = InterferometerList(ifos)
        # compute merger times in ifos (according to theta)
        time_event = dingo_result.event_metadata["time_event"]
        time_ifos = {}
        for ifo in ifo_list:
            dt = time_delay_from_geocenter(ifo, theta["ra"], theta["dec"], time_event)
            time_ifos[ifo.name] = time_event + theta["geocent_time"] + dt

        # generate single-detector event objects
        singles = []
        for ifo in ifo_list:
            single = DingoSingleEvent(
                ifo.name,
                time_event,
                data=dingo_result.context["waveform"][ifo.name],
                asd=dingo_result.context["asds"][ifo.name],
                template=template,
                domain=domain,
                duration=duration,
                f_lower=f_lower,
                f_upper=f_upper,
                t_center=time_ifos[ifo.name],
                t_width=t_search_window_width,
            )
            singles.append(single)

        merger_times_theta = {k: np.round(v, 3) for k, v in time_ifos.items()}
        print(f"Merger times according to theta: {merger_times_theta}")
        merger_times_snr = {
            ifo.name: np.round(single.time, 3) for ifo, single in zip(ifo_list, singles)
        }
        print(f"Merger times according to snr series: {merger_times_snr}")

        return cls(singles=singles, template_args=theta)


if __name__ == "__main__":
    dingo_result = Result(
        file_name="/Users/maxdax/Documents/Projects/GW-Inference/01_bns/results/01_real_data/01_GWTC/01_data/phase_marginalization_22/GW170817_lowSpin.hdf5"
    )
    event = DingoEvent.from_dingo_result(
        dingo_result,
        max_likelihood_template=True,
        duration=0.1,
        f_lower=23,
        f_upper=1024,
        t_search_window_width=0.25,
    )
