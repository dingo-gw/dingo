"""Gravitational-wave chain steps: the GNPE factors, the synthetic-phase
factor, and the coordinate reparametrizations."""

from __future__ import annotations
import logging
import time
from typing import Optional
import numpy as np
import pandas as pd
import torch
from astropy.time import Time
from bilby.core.prior import PriorDict
from bilby.gw.detector import InterferometerList
from torchvision.transforms import Compose
from dingo.core.density import (
    interpolated_log_prob_multi,
    interpolated_sample_and_log_prob_multi,
)
from dingo.core.factors import (
    Factor,
    Reparametrization,
    Standardization,
    TargetCorrection,
    _base_model_metadata,
    _n_rows,
)
from dingo.core.multiprocessing import apply_func_with_multiprocessing
from dingo.core.posterior_models import BasePosteriorModel
from dingo.core.transforms import RenameKey
from dingo.gw.conversion import change_spin_conversion_phase
from dingo.gw.domains import build_domain
from dingo.gw.transforms import (
    CopyToExtrinsicParameters,
    GetDetectorTimes,
    GNPEBase,
    GNPECoalescenceTimes,
    PostCorrectGeocentTime,
    SelectStandardizeRepackageParameters,
    TimeShiftStrain,
)

logger = logging.getLogger(__name__)


def _to_numpy(v) -> np.ndarray:
    """Detach a torch tensor (or coerce anything) to a numpy array."""
    if torch.is_tensor(v):
        return v.detach().cpu().numpy()
    return np.asarray(v)


class SyntheticPhaseFactor(Factor):
    """
    Reconstruct the coalescence phase for a phase-marginalized network:
    `q(phase | theta_rest, d)` from the likelihood on a phase grid. The terminal factor of
    the chain.

    For each incoming `theta_rest` it builds the phase-full likelihood
    (`context.likelihood()`) and evaluates `log L` on a grid over `[0, 2 pi)`, exploiting
    that the waveform modes computed once at `phase = 0` each transform as `exp(-i m phase)`
    -- so the whole grid follows from a single waveform evaluation. The grid is
    exponentiated into a conditional phase distribution, a uniform floor (weight
    `uniform_weight`) is added to keep it mass-covering, and one phase is drawn per sample
    from the interpolated distribution. The returned proposal log-prob
    `log q(phase | theta_rest, d)` joins the chain's proposal density; importance sampling
    then targets the phase-full posterior (with the phase prior re-added).

    Two grid modes: `approximation_22_mode=True` assumes a (2, 2)-dominated signal (the
    whole waveform transforms as `exp(2i phase)`), giving `log L` from the complex overlap
    `Re[(d | h(phase=0)) exp(2i phase)]`; `False` sums the modes exactly and requires the
    waveform generator's `spin_conversion_phase = 0`. Note the entry points differ on the
    default: this factor and `dingo_pipe`'s `PhaseRecoveryDefault` use the exact mode,
    while `Result.sample_synthetic_phase` defaults to the (2, 2) approximation when the
    key is omitted.
    """

    def __init__(
        self,
        conditioning: list[str],
        n_grid: int = 5001,
        approximation_22_mode: bool = False,
        uniform_weight: float = 0.01,
        num_processes: int = 1,
    ):
        """
        Parameters
        ----------
        conditioning : list[str]
            The physical parameters the likelihood needs to generate the waveform
            (everything the chain has produced except `phase`).
        n_grid : int, default 5001
            Number of phase grid points on `[0, 2 pi)`.
        approximation_22_mode : bool, default False
            Use the (2, 2)-mode approximation instead of the exact mode sum.
        uniform_weight : float, default 0.01
            Weight of the uniform floor added to the phase distribution for mass coverage.
        num_processes : int, default 1
            Parallel processes for the per-sample likelihood evaluation and phase sampling.
        """
        self.parameters = ["phase"]
        self.conditioning = list(conditioning)
        self.n_grid = n_grid
        self.approximation_22_mode = approximation_22_mode
        self.uniform_weight = uniform_weight
        self.num_processes = num_processes

    def sample_and_log_prob(self, num_samples, context, given=None):
        """Draw one phase per `theta_rest` row (`num_samples` must be 1); return the phases
        and their proposal log-prob `log q(phase | theta_rest, d)`."""
        if num_samples != 1:
            raise ValueError(
                "Synthetic phase is 1:1; draw one phase per sample (fan_out=1)."
            )
        reference = next(iter(given.values()))
        device = reference.device if torch.is_tensor(reference) else None
        n = len(reference)
        logger.info(f"Estimating synthetic phase for {n} samples.")
        t0 = time.time()
        phases, phase_posterior = self._phase_profile(given, context)
        new_phase, log_prob = interpolated_sample_and_log_prob_multi(
            phases, phase_posterior, self.num_processes
        )
        logger.info(f"Done. This took {time.time() - t0:.2f} s.")
        return (
            {"phase": torch.as_tensor(new_phase, device=device)},
            torch.as_tensor(log_prob, device=device),
        )

    def log_prob(self, theta_i, context, given=None):
        """Evaluate `log q(phase | theta_rest, d)` at the given phases (re-plug / IS)."""
        reference = next(iter(given.values()))
        device = reference.device if torch.is_tensor(reference) else None
        phases, phase_posterior = self._phase_profile(given, context)
        log_prob = interpolated_log_prob_multi(
            phases, phase_posterior, _to_numpy(theta_i["phase"]), self.num_processes
        )
        return torch.as_tensor(log_prob, device=device)

    def describe(self) -> dict:
        return {
            "step": type(self).__name__,
            "parameters": list(self.parameters),
            "conditioning": list(self.conditioning),
            "n_grid": self.n_grid,
            "approximation_22_mode": self.approximation_22_mode,
            "uniform_weight": self.uniform_weight,
        }

    def _phase_profile(self, given, context):
        """The phase grid and the mass-covered (un-normalized) phase distribution, one row
        per sample: evaluate `log L` on the grid, exponentiate (shifted by the per-row
        max), and add the uniform floor."""
        theta = pd.DataFrame({k: _to_numpy(v) for k, v in given.items()})
        # The context (possibly derived for importance sampling) carries the
        # representation; the phase-full likelihood needs no arguments.
        likelihood = context.likelihood()
        phases = np.linspace(0, 2 * np.pi, self.n_grid)
        if self.approximation_22_mode:
            # Assume the waveform is (2, 2)-dominated (transforms as exp(2i phase)), so the
            # phase-dependent log-posterior is Re[(d | h(phase=0)) exp(2i phase)].
            theta = theta.copy()
            theta["phase"] = 0.0
            d_inner_h = likelihood.d_inner_h_complex_multi(theta, self.num_processes)
            phase_log_posterior = np.outer(d_inner_h, np.exp(2j * phases)).real
        else:
            # Exact: each mode m contributes exp(-i m phase); needs spin_conversion_phase=0.
            likelihood.phase_grid = phases
            phase_log_posterior = apply_func_with_multiprocessing(
                likelihood.log_likelihood_phase_grid,
                theta,
                num_processes=self.num_processes,
            )
        phase_posterior = np.exp(
            phase_log_posterior - np.amax(phase_log_posterior, axis=1, keepdims=True)
        )
        # Uniform floor: keep q(phase) > 0 everywhere so importance sampling stays finite.
        phase_posterior += (
            phase_posterior.mean(axis=-1, keepdims=True) * self.uniform_weight
        )
        return phases, phase_posterior


def _build_gnpe_transforms(model: BasePosteriorModel):
    """Build the time-shift GNPE per-step transforms from a model's metadata: the
    proxy blur, the per-row time-shift alignment applied before the network, and the
    post-network geocent-time correction.

    Returns
    -------
    transform_pre, transform_post : Compose
    gnpe_parameters : list[str]
        The GNPE input parameters (detector times).
    inference_parameters : list[str]
    kernel : PriorDict
        The proxy perturbation kernel.
    gnpe_transform : GNPECoalescenceTimes
        The blur transform itself, shared so the kernel factor can call `sample_proxies`.
    """
    meta = _base_model_metadata(model)
    data_settings = meta["train_settings"]["data"]
    ifo_list = InterferometerList(data_settings["detectors"])
    domain = build_domain(meta["dataset_settings"]["domain"])
    if "domain_update" in data_settings:
        domain.update(data_settings["domain_update"])

    gnpe_time_settings = data_settings.get("gnpe_time_shifts")
    if not gnpe_time_settings:
        raise NotImplementedError(
            "Only time-shift GNPE is supported here so far (no gnpe_chirp / gnpe_phase)."
        )

    gnpe_transform = GNPECoalescenceTimes(
        ifo_list,
        gnpe_time_settings["kernel"],
        gnpe_time_settings["exact_equiv"],
        inference=True,
    )
    transform_pre = [
        RenameKey("data", "waveform"),
        gnpe_transform,
        TimeShiftStrain(ifo_list, domain),
        SelectStandardizeRepackageParameters(
            {"context_parameters": data_settings["context_parameters"]},
            data_settings["standardization"],
            device=model.device,
        ),
        RenameKey("waveform", "data"),
    ]
    gnpe_parameters: list[str] = []
    kernel = PriorDict()
    for transform in transform_pre:
        if isinstance(transform, GNPEBase):
            gnpe_parameters += transform.input_parameter_names
            for k, v in transform.kernel.items():
                kernel[k] = v

    inference_parameters = data_settings["inference_parameters"]
    transform_post = [
        SelectStandardizeRepackageParameters(
            {"inference_parameters": inference_parameters},
            data_settings["standardization"],
            inverse=True,
            as_type="dict",
        ),
        PostCorrectGeocentTime(),
        CopyToExtrinsicParameters(
            "ra", "dec", "geocent_time", "chirp_mass", "mass_ratio", "phase"
        ),
        GetDetectorTimes(ifo_list, data_settings["ref_time"]),
    ]
    return (
        Compose(transform_pre),
        Compose(transform_post),
        gnpe_parameters,
        inference_parameters,
        kernel,
        gnpe_transform,
    )


class GNPEKernelFactor(Factor):
    """
    The GNPE perturbation kernel `p(theta_hat | theta)` as a non-network factor.

    `theta` are the detector coalescence times; the kernel adds a bounded perturbation to
    each, giving the proxies `theta_hat` the main network conditions on. The parameter
    block is the proxies, the conditioning is the detector times.
    `sample_and_log_prob` blurs the times into proxies (the proxy update of a Gibbs
    sweep); `log_prob` returns the kernel density `log p(theta_hat | theta)` at the
    proxies and the detector times. One proxy per detector-time row.
    """

    def __init__(self, gnpe_transform: GNPEBase, gnpe_parameters: list[str]):
        """
        Parameters
        ----------
        gnpe_transform : GNPEBase
            The blur transform supplying the kernel and `sample_proxies`.
        gnpe_parameters : list[str]
            The detector-time parameters perturbed into proxies.
        """
        self.gnpe = gnpe_transform
        self.gnpe_parameters = gnpe_parameters
        self.parameters = [p + "_proxy" for p in gnpe_parameters]
        self.conditioning = list(gnpe_parameters)
        self.kernel = gnpe_transform.kernel

    @classmethod
    def from_model(cls, model: BasePosteriorModel) -> "GNPEKernelFactor":
        """Build from the main model's metadata (the kernel / blur transform)."""
        _, _, gnpe_parameters, _, _, gnpe_transform = _build_gnpe_transforms(model)
        return cls(gnpe_transform, gnpe_parameters)

    def sample_and_log_prob(self, num_samples, context, given=None):
        """Blur the conditioning detector times into proxies; `num_samples` must be 1
        (GNPE is 1:1). Returns the proxies and their kernel log-prob."""
        if num_samples != 1:
            raise ValueError("GNPE proxy is 1:1; use fan_out=1.")
        times = {k: given[k] for k in self.gnpe_parameters}
        proxies = self.gnpe.sample_proxies(times)
        return proxies, self.log_prob(proxies, context, given)

    def log_prob(self, theta_i, context, given=None):
        """`log p(theta_hat | theta)` from the kernel, at the proxies (`theta_i`) and
        the detector times (`given`).

        The kernel is a bilby `PriorDict` -- the same object that samples the blur --
        so the density is evaluated in numpy (converting each side first: the times
        and proxies may live on different devices) and returned on the detector
        times' device."""
        reference = next(iter(given.values()))
        device = reference.device if torch.is_tensor(reference) else None
        diffs = {
            k: _to_numpy(given[k]) - _to_numpy(theta_i[f"{k}_proxy"])
            for k in self.kernel.keys()
        }
        return torch.as_tensor(
            self.kernel.ln_prob(diffs, axis=0), dtype=torch.float32, device=device
        )


class GNPEFlowFactor(Factor):
    """
    The GNPE main network `q(theta | theta_hat, d)` as a factor.

    Conditions on the detector-time proxies from `GNPEKernelFactor`: it shifts each
    detector's strain by the corresponding proxy time (standardizing the network input),
    samples the network, and recomputes the detector times from the sampled sky position
    and geocent time. The proxies are supplied, so no blurring happens here.

    The single network factor in either GNPE mode: cycled by a `GibbsBlock` for
    multi-iteration GNPE, or a `ChainComposer` factor for single-step GNPE. The recomputed
    detector times are emitted as extra columns (`produces`): the next Gibbs iteration
    blurs them into fresh proxies, and single-step GNPE evaluates the kernel correction at
    them. One sample per proxy row.
    """

    def __init__(
        self,
        model: BasePosteriorModel,
        transform_pre: Compose,
        transform_post: Compose,
        gnpe_parameters: list[str],
        parameters: list[str],
        aliases: Optional[dict[str, str]] = None,
    ):
        """
        Parameters
        ----------
        model : BasePosteriorModel
            The GNPE main network.
        transform_pre : Compose
            Per-iteration pre-network transforms (proxy bookkeeping, time shift,
            standardization).
        transform_post : Compose
            Post-network transforms (de-standardize, recompute detector times).
        gnpe_parameters : list[str]
            The detector-time parameters.
        parameters : list[str]
            The network's trained inference parameters.
        aliases : dict[str, str], optional
            Trained-name to exposed-name map (e.g. `{"ra": "ra@t_ref"}`).
        """
        self.model = model
        self.transform_pre = transform_pre
        self.transform_post = transform_post
        self.gnpe_parameters = gnpe_parameters
        self.proxy_parameters = [p + "_proxy" for p in gnpe_parameters]
        self.aliases = aliases or {}
        self._net_parameters = parameters
        self.parameters = [self.aliases.get(p, p) for p in parameters]
        self.conditioning = list(self.proxy_parameters)
        # For log_prob: the sampling path de-standardizes (and corrects the log-prob)
        # inside transform_post, but evaluating at a point needs the forward map too.
        # The model's own standardization (network-bound, like FlowFactor's).
        std = model.metadata["train_settings"]["data"]["standardization"]
        self.standardization = Standardization(std["mean"], std["std"])

    @property
    def produces(self) -> list[str]:
        """Emitted columns: the inference block plus the recomputed detector times."""
        return self.parameters + self.gnpe_parameters

    @classmethod
    def from_model(
        cls, model: BasePosteriorModel, aliases: Optional[dict[str, str]] = None
    ) -> "GNPEFlowFactor":
        """Build the GNPE per-iteration transforms from the main model's metadata.

        Parameters
        ----------
        model : BasePosteriorModel
            The GNPE main model.
        aliases : dict[str, str], optional
            Trained-name to canonical-name map (e.g. `{"ra": "ra@t_ref"}`).

        Returns
        -------
        GNPEFlowFactor
        """
        pre, post, gnpe_parameters, inference_parameters, _, _ = _build_gnpe_transforms(
            model
        )
        return cls(
            model, pre, post, gnpe_parameters, inference_parameters, aliases=aliases
        )

    def sample_and_log_prob(self, num_samples, context, given=None):
        """Sample one parameter set per proxy row; `num_samples` must be 1 (GNPE is 1:1).
        Returns theta plus the recomputed detector times, and the network log-prob."""
        if num_samples != 1:
            raise ValueError("GNPE is 1:1; draw one sample per proxy (fan_out=1).")
        proxies = {p: given[p] for p in self.proxy_parameters}
        n_rows = _n_rows(proxies)
        x = {"extrinsic_parameters": dict(proxies), "parameters": {}}
        d = context.prepared_data().clone()
        x["data"] = d.expand(n_rows, *d.shape)
        x = self.transform_pre(x)
        self.model.network.eval()
        with torch.no_grad():
            if "context_parameters" in x:
                y, log_prob = self.model.sample_and_log_prob(
                    x["data"], x["context_parameters"]
                )
            else:
                y, log_prob = self.model.sample_and_log_prob(x["data"])
        # sample_and_log_prob(num_samples=1) adds a singleton dim; the batch is the proxy
        # rows.
        x["parameters"] = y.squeeze(1)
        x["log_prob"] = log_prob.squeeze(1)
        x = self.transform_post(x)
        params = dict(x["parameters"])
        # Expose trained names under their canonical aliases (e.g. ra -> ra@t_ref).
        params = {self.aliases.get(k, k): v for k, v in params.items()}
        # Surface the recomputed detector times: the next Gibbs iteration's input, and the
        # evaluation point for GNPEKernelFactor's importance-sampling correction.
        for k in self.gnpe_parameters:
            params[k] = x["extrinsic_parameters"][k]
        return params, x["log_prob"]

    def log_prob(self, theta_i, context, given=None):
        """Evaluate the network density `log q(theta | theta_hat, d)` in physical space
        at given `theta_i` (exposed / aliased names), one row per proxy row in `given`.
        Applies the same per-row view of the data as sampling (the shared
        representation time-shifted by the proxies, the conditioning standardized),
        then scores the standardized parameters under the network."""
        proxies = {p: given[p] for p in self.proxy_parameters}
        n_rows = _n_rows(proxies)
        x = {"extrinsic_parameters": dict(proxies), "parameters": {}}
        d = context.prepared_data().clone()
        x["data"] = d.expand(n_rows, *d.shape)
        x = self.transform_pre(x)
        theta_net = {
            net: theta_i[self.aliases.get(net, net)] for net in self._net_parameters
        }
        # Mirror transform_post: sampling shifts geocent_time by the preferred proxy
        # after the network (PostCorrectGeocentTime, using the extrinsic geocent_time
        # set up by the GNPE transform), so score the network in its own output frame
        # by applying the inverse correction first.
        y = {
            "parameters": dict(theta_net),
            "extrinsic_parameters": dict(x["extrinsic_parameters"]),
        }
        theta_net = PostCorrectGeocentTime(inverse=True)(y)["parameters"]
        z = self.standardization.standardize(theta_net, self._net_parameters)
        self.model.network.eval()
        with torch.no_grad():
            if "context_parameters" in x:
                log_prob = self.model.log_prob(z, x["data"], x["context_parameters"])
            else:
                log_prob = self.model.log_prob(z, x["data"])
        return log_prob + self.standardization.log_det(self._net_parameters)


class RAToEventFrame(Reparametrization):
    """
    Rotate right ascension from the network's training reference frame (`ra@t_ref`) to the
    event frame (`ra`).

    The network is trained at a fixed reference time; an event at a different GPS time needs
    the sky rotated by the sidereal-time difference. This is a measure-preserving shift
    modulo 2*pi (`log_det = 0`), so it contributes nothing to the density. `forward`
    produces the event-frame `ra`, `inverse` recovers `ra@t_ref`. The sidereal
    correction is read from the shared context (`t_ref` and the event time).

    The modulo makes the map a bijection on the circle, while the flow's density lives
    on the real line: a sample drawn outside `[0, 2 pi)` is wrapped, so `inverse`
    recovers its principal-branch representative and a re-evaluated `log_prob` refers
    to that branch. Only tail samples outside the bounded `ra` prior are affected.
    """

    def __init__(self):
        self.conditioning = ["ra@t_ref"]
        self.parameters = ["ra"]

    @staticmethod
    def _correction(context) -> float:
        """Sidereal-time difference (event minus reference) in radians; 0 when the event
        time is unset or equal to the reference time."""
        event_metadata = context.event_metadata
        t_event = None if event_metadata is None else event_metadata.get("time_event")
        t_ref = context.t_ref
        if t_event is None or t_event == t_ref:
            return 0.0
        longitude_event = Time(t_event, format="gps", scale="utc").sidereal_time(
            "apparent", "greenwich"
        )
        longitude_reference = Time(t_ref, format="gps", scale="utc").sidereal_time(
            "apparent", "greenwich"
        )
        return (longitude_event - longitude_reference).rad

    def forward(self, given, context):
        correction = self._correction(context)
        if correction == 0.0:
            return {"ra": given["ra@t_ref"]}
        # ra is a bounded angle -> float32 is plenty. The correction is a difference of
        # absolute GPS times, so compute it in float64, but store the wrapped angle float32.
        ra = (given["ra@t_ref"].double() + correction) % (2 * np.pi)
        return {"ra": ra.float()}

    def inverse(self, params, context, given=None):
        correction = self._correction(context)
        if correction == 0.0:
            return {"ra@t_ref": params["ra"]}
        ra_tref = (params["ra"].double() - correction) % (2 * np.pi)
        return {"ra@t_ref": ra_tref.float()}


class RAToTrainingFrame(RAToEventFrame):
    """
    Rotate a pinned event-frame right ascension (`ra`) into the network's training
    reference frame (`ra@t_ref`) before it conditions the network: the input-side
    mirror of `RAToEventFrame`. A parameter that is frame-corrected on the output side
    must be inversely corrected on the input side, so a sky position pinned at the
    event time is presented to the network in the frame it was trained in; a
    trailing `RAToEventFrame` restores the event-frame value in the samples.
    """

    def __init__(self):
        self.conditioning = ["ra"]
        self.parameters = ["ra@t_ref"]

    def forward(self, given, context):
        return super().inverse({"ra": given["ra"]}, context)

    def inverse(self, params, context, given=None):
        return super().forward({"ra@t_ref": params["ra@t_ref"]}, context)


class SpinConventionReparam(Reparametrization):
    """
    Relabel the precessing-spin angles between Dingo's internal spin-phase
    convention and the physical (Bilby) one.

    Dingo fixes the spin-conversion phase (usually to 0) so that the Cartesian
    spins decouple from the coalescence phase; the whole density / likelihood /
    synthetic-phase pipeline requires that convention, so stored samples keep the
    plain names `theta_jn` / `phi_jl` in the *network* convention throughout. The
    physical convention (spin conversion at the sample's own phase) is what Bilby
    and PESummary mean by the same names, so the relabel happens at the export
    boundary -- this class is its single home. Only `theta_jn` and `phi_jl`
    change; the conversion phase and reference frequency are read from the model
    metadata, and a model trained without a fixed conversion phase (`None`)
    relabels to the identity.

    Unlike `RAToEventFrame` no marked intermediate name is needed: the two conventions
    never coexist in one table -- each world's plain names denote its own
    convention, and this bijection is the boundary crossing.

    Exporting a finished weighted sample set needs no Jacobian (proposal, prior,
    and likelihood transform together), which is how `to_physical` is used. As a
    chain `Step` the bijection is *not* measure-preserving in the flat
    `(theta_jn, phi_jl)` coordinates: it rotates the line of sight rigidly about
    the orbital angular momentum, preserving the spherical measure
    `sin(theta_jn) dtheta dphi`, so
    `log_det = log sin(theta_jn) - log sin(theta_jn')` (verified numerically
    against finite differences through the LAL conversion), and `inverse`
    rebuilds the network convention from the physical one using the invariant
    conditioning the reverse fold supplies.
    """

    def __init__(self, num_processes: int = 1):
        """
        Parameters
        ----------
        num_processes : int, default 1
            Parallel processes for the per-sample LAL spin conversion.
        """
        self.parameters = ["theta_jn", "phi_jl"]
        self.conditioning = [
            "theta_jn",
            "phi_jl",
            "phase",
            "chirp_mass",
            "mass_ratio",
            "a_1",
            "a_2",
            "tilt_1",
            "tilt_2",
            "phi_12",
        ]
        self.num_processes = num_processes

    @property
    def consumes(self) -> list[str]:
        # The bijection overwrites theta_jn / phi_jl in place; the remaining
        # conditioning (phase, masses, tilts, ...) is read-only and must stay in
        # the chain -- the default conditioning-minus-parameters would drop it.
        return []

    def log_det(self, given, context):
        """`log|det J|` of `forward`, per row. The map preserves the spherical
        measure, so the flat-coordinate Jacobian is
        `sin(theta_jn) / sin(theta_jn')` -- verified numerically against finite
        differences through the LAL conversion in
        `test_jacobian_matches_sin_ratio` (agreement ~1e-9)."""
        converted = self.forward(given, context)
        return self._log_det(given["theta_jn"], converted["theta_jn"])

    @staticmethod
    def _log_det(theta_jn_in, theta_jn_out):
        # Compute in double (the LAL conversion is double precision anyway),
        # return in the input dtype: a reparametrization preserves the chain's
        # dtype rather than promoting the summed log_prob.
        log_det = torch.log(torch.sin(theta_jn_in.double())) - torch.log(
            torch.sin(theta_jn_out.double())
        )
        return log_det.to(theta_jn_in.dtype)

    def sample_and_log_prob(self, num_samples, context, given=None):
        """Apply `forward`; contribute `-log|det J|`. Overridden to share the
        single LAL conversion between the transform and its Jacobian (the base
        implementation would convert twice)."""
        if num_samples != 1:
            raise ValueError("A reparametrization is 1:1; use fan_out=1.")
        out = self.forward(given, context)
        return out, -self._log_det(given["theta_jn"], out["theta_jn"])

    @staticmethod
    def _model_convention(model_metadata: dict) -> tuple[float, Optional[float]]:
        """The reference frequency and spin-conversion phase the model trained with."""
        wfg_settings = model_metadata["dataset_settings"]["waveform_generator"]
        return wfg_settings["f_ref"], wfg_settings.get("spin_conversion_phase")

    def to_physical(self, samples: pd.DataFrame, model_metadata: dict) -> pd.DataFrame:
        """Relabel samples from the model's convention to the physical (Bilby) one."""
        f_ref, sc_phase = self._model_convention(model_metadata)
        return change_spin_conversion_phase(
            samples, f_ref, sc_phase, None, num_processes=self.num_processes
        )

    def to_network(self, samples: pd.DataFrame, model_metadata: dict) -> pd.DataFrame:
        """Relabel samples from the physical (Bilby) convention to the model's,
        e.g. to ingest external posteriors for comparison."""
        f_ref, sc_phase = self._model_convention(model_metadata)
        return change_spin_conversion_phase(
            samples, f_ref, None, sc_phase, num_processes=self.num_processes
        )

    def forward(self, given, context):
        # The conversion runs in double; the outputs return in the input dtype
        # and device (cf. RAToEventFrame: compute in float64, store the chain dtype).
        reference = given["theta_jn"]
        theta = pd.DataFrame({k: _to_numpy(v) for k, v in given.items()})
        converted = self.to_physical(theta, context.model_metadata)
        return {
            k: torch.as_tensor(converted[k].to_numpy()).to(
                dtype=reference.dtype, device=reference.device
            )
            for k in self.parameters
        }

    def inverse(self, params, context, given=None):
        # The physical -> network direction also needs the invariant
        # conditioning (phase, masses, tilts), which the reverse fold supplies
        # as `given`.
        if given is None:
            raise ValueError(
                "The spin-convention inverse needs the conditioning block "
                "(phase, masses, tilts); pass it as `given`, or convert "
                "DataFrames with to_network()."
            )
        reference = params["theta_jn"]
        rows = {**given, **params}
        theta = pd.DataFrame({k: _to_numpy(v) for k, v in rows.items()})
        converted = self.to_network(theta, context.model_metadata)
        return {
            k: torch.as_tensor(converted[k].to_numpy()).to(
                dtype=reference.dtype, device=reference.device
            )
            for k in self.parameters
        }


class GNPEKernelCorrection(TargetCorrection):
    """
    The single-step GNPE kernel correction as a target-side chain step.

    Emits `delta_log_prob_target = log p(theta_hat | theta)` -- the GNPE kernel evaluated
    at the proxies and the detector times recomputed from theta -- for importance sampling
    on the joint proposal `q(theta, theta_hat | d)`. Contributes 0 to the proposal
    density and consumes the intermediate detector times.
    """

    def __init__(self, kernel_factor: GNPEKernelFactor):
        self.kernel_factor = kernel_factor
        self.parameters = ["delta_log_prob_target"]
        self.conditioning = list(kernel_factor.parameters) + list(
            kernel_factor.gnpe_parameters
        )
        self.consumes = list(kernel_factor.gnpe_parameters)

    def correction(self, given, context):
        proxies = {p: given[p] for p in self.kernel_factor.parameters}
        times = {k: given[k] for k in self.kernel_factor.gnpe_parameters}
        correction = self.kernel_factor.log_prob(proxies, context, times)
        return {"delta_log_prob_target": correction}
