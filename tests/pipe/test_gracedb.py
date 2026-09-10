"""Tests for dingo.pipe.gracedb (dingo_pipe_gracedb).

All tests are self-contained: no GraceDB network access, no real dingo model,
and no CIT cluster (calibration lookup fails gracefully off-cluster).
"""

import copy
import json
import os
from pathlib import Path

import pytest

from dingo.pipe.gracedb import (
    _check_model_compatibility,
    _extract_prior_from_metadata,
    _extract_prior_from_model,
    _find_init_model,
    _get_analysis_duration,
    _is_gnpe_model,
    _write_config_file,
    create_parser,
    prepare_dingo_config,
)

# ---------------------------------------------------------------------------
# Mock event data
# ---------------------------------------------------------------------------

MOCK_CANDIDATE_BBH = {
    "graceid": "T123456a",
    "superevent": "TS123456a",
    "group": "CBC",
    "gpstime": 1234567890.0,
    "coinc_file": None,
    "extra_attributes": {
        "CoincInspiral": {"ifos": "H1,L1", "mchirp": 28.3},
        "SingleInspiral": [
            {
                "ifo": "H1",
                "snr": 12.5,
                "mchirp": 28.3,
                "mass1": 50.0,
                "mass2": 30.0,
                "spin1z": 0.1,
                "spin2z": 0.05,
                "end_time": 1234567890,
                "end_time_ns": 0,
            },
            {
                "ifo": "L1",
                "snr": 10.0,
                "mchirp": 28.3,
                "mass1": 50.0,
                "mass2": 30.0,
                "spin1z": 0.1,
                "spin2z": 0.05,
                "end_time": 1234567890,
                "end_time_ns": 100_000_000,
            },
        ],
    },
}

MOCK_CANDIDATE_BNS = {
    **MOCK_CANDIDATE_BBH,
    "graceid": "T654321b",
    "superevent": "TS654321b",
    "extra_attributes": {
        "CoincInspiral": {"ifos": "H1,L1", "mchirp": 1.2},
        "SingleInspiral": [
            {
                "ifo": "H1",
                "snr": 15.0,
                "mchirp": 1.2,
                "mass1": 1.4,
                "mass2": 1.2,
                "spin1z": 0.01,
                "spin2z": 0.01,
                "end_time": 1234567890,
                "end_time_ns": 0,
            },
            {
                "ifo": "L1",
                "snr": 12.0,
                "mchirp": 1.2,
                "mass1": 1.4,
                "mass2": 1.2,
                "spin1z": 0.01,
                "spin2z": 0.01,
                "end_time": 1234567890,
                "end_time_ns": 50_000_000,
            },
        ],
    },
}

# ---------------------------------------------------------------------------
# Mock model metadata
# ---------------------------------------------------------------------------

# Represents an 8-second BBH model (delta_f = 0.125 → duration = 8s).
MOCK_MODEL_METADATA = {
    "dataset_settings": {
        "domain": {
            "type": "FrequencyDomain",
            "f_min": 20.0,
            "f_max": 1024.0,
            "delta_f": 0.125,  # → 8s
        },
        "intrinsic_prior": {
            "mass_1": "bilby.core.prior.Constraint(minimum=10.0, maximum=120.0)",
            "mass_2": "bilby.core.prior.Constraint(minimum=10.0, maximum=120.0)",
            "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=20.0, maximum=120.0)",
            "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=1.0)",
            "phase": "default",
            "a_1": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)",
            "a_2": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)",
            "tilt_1": "default",
            "tilt_2": "default",
            "phi_12": "default",
            "phi_jl": "default",
            "theta_jn": "default",
            # Fixed scalar values – NOT priors, must be excluded
            "luminosity_distance": 100.0,
            "geocent_time": 0.0,
        },
    },
    "train_settings": {
        "data": {
            "extrinsic_prior": {
                "dec": "default",
                "ra": "default",
                "geocent_time": "bilby.core.prior.Uniform(minimum=-0.10, maximum=0.10)",
                "psi": "default",
                "luminosity_distance": "bilby.core.prior.Uniform(minimum=100.0, maximum=1000.0)",
            },
        },
    },
}

MOCK_PRIOR_DICT = {
    "mass_1": "bilby.core.prior.Constraint(minimum=10.0, maximum=120.0)",
    "mass_2": "bilby.core.prior.Constraint(minimum=10.0, maximum=120.0)",
    "chirp_mass": "bilby.gw.prior.UniformInComponentsChirpMass(minimum=20.0, maximum=120.0)",
    "mass_ratio": "bilby.gw.prior.UniformInComponentsMassRatio(minimum=0.125, maximum=1.0)",
    "phase": "default",
    "a_1": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)",
    "a_2": "bilby.core.prior.Uniform(minimum=0.0, maximum=0.99)",
    "tilt_1": "default",
    "tilt_2": "default",
    "phi_12": "default",
    "phi_jl": "default",
    "theta_jn": "default",
    "dec": "default",
    "ra": "default",
    "geocent_time": "bilby.core.prior.Uniform(minimum=-0.10, maximum=0.10)",
    "psi": "default",
    "luminosity_distance": "bilby.core.prior.Uniform(minimum=100.0, maximum=1000.0)",
}


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_model(tmp_path):
    p = tmp_path / "model.pt"
    p.write_bytes(b"fake model content")
    return str(p)


@pytest.fixture
def candidate_json(tmp_path):
    p = tmp_path / "T123456a.json"
    p.write_text(json.dumps(MOCK_CANDIDATE_BBH))
    return str(p)


@pytest.fixture
def mock_metadata(monkeypatch):
    """Patch _load_model_metadata to return the mock 8-second BBH metadata."""
    import dingo.pipe.gracedb as gracedb_mod
    monkeypatch.setattr(
        gracedb_mod, "_load_model_metadata", lambda _: MOCK_MODEL_METADATA
    )


# Same 8-second BBH model, but trained with GNPE time shifts — such a model
# requires a second init model at inference time.
MOCK_GNPE_MODEL_METADATA = copy.deepcopy(MOCK_MODEL_METADATA)
MOCK_GNPE_MODEL_METADATA["train_settings"]["data"]["gnpe_time_shifts"] = {
    "kernel": "bilby.core.prior.Uniform(minimum=-0.001, maximum=0.001)",
    "exact_equiv": True,
}


@pytest.fixture
def mock_gnpe_metadata(monkeypatch):
    """Patch _load_model_metadata to return GNPE model metadata."""
    import dingo.pipe.gracedb as gracedb_mod
    monkeypatch.setattr(
        gracedb_mod, "_load_model_metadata", lambda _: MOCK_GNPE_MODEL_METADATA
    )


# ---------------------------------------------------------------------------
# _get_analysis_duration
# ---------------------------------------------------------------------------


class TestGetAnalysisDuration:
    def test_high_mass_bbh(self):
        assert _get_analysis_duration(30) == 4

    def test_intermediate_mass(self):
        assert _get_analysis_duration(10) == 8

    def test_nsbh(self):
        assert _get_analysis_duration(4.0) == 32

    def test_bns(self):
        assert _get_analysis_duration(1.2) == 128

    def test_boundary_above_13_53(self):
        assert _get_analysis_duration(14.0) == 4

    def test_boundary_below_13_53(self):
        assert _get_analysis_duration(13.0) == 8


# ---------------------------------------------------------------------------
# _check_model_compatibility
# ---------------------------------------------------------------------------


class TestCheckModelCompatibility:
    # Re-use the same metadata structure with explicit durations for clarity.

    def _metadata(self, delta_f):
        """Build minimal metadata with the given delta_f (= 1/duration)."""
        import copy
        m = copy.deepcopy(MOCK_MODEL_METADATA)
        m["dataset_settings"]["domain"]["delta_f"] = delta_f
        return m

    def _mfd_metadata(self, base_delta_f):
        """Build metadata for a MultibandedFrequencyDomain model.

        Production models use an MFD whose delta_f lives under base_domain, not
        at the top level (duration = 1 / base_domain.delta_f).
        """
        import copy
        m = copy.deepcopy(MOCK_MODEL_METADATA)
        m["dataset_settings"]["domain"] = {
            "type": "MultibandedFrequencyDomain",
            "base_domain": {
                "type": "FrequencyDomain",
                "f_min": 20.0,
                "f_max": 1024.0,
                "delta_f": base_delta_f,
            },
            "nodes": [20.0, 35.0, 63.5, 97.5, 1023.5],
            "delta_f_initial": base_delta_f,
        }
        return m

    def test_compatible_bbh_within_model_duration(self):
        # chirp_mass=28.3 → 4s required; model is 8s → OK
        _check_model_compatibility(28.3, self._metadata(0.125))

    def test_compatible_equal_duration(self):
        # chirp_mass=10 → 8s required; model is exactly 8s → OK
        _check_model_compatibility(10.0, self._metadata(0.125))

    def test_incompatible_bns_raises(self):
        # chirp_mass=1.2 → 128s required; model is 8s → error
        with pytest.raises(ValueError, match="BNS or NSBH"):
            _check_model_compatibility(1.2, self._metadata(0.125))

    def test_incompatible_nsbh_raises(self):
        # chirp_mass=2.5 → 64s required; model is 8s → error
        with pytest.raises(ValueError, match="64s"):
            _check_model_compatibility(2.5, self._metadata(0.125))

    def test_error_message_includes_model_duration(self):
        with pytest.raises(ValueError, match="8s"):
            _check_model_compatibility(1.2, self._metadata(0.125))

    def test_error_message_includes_chirp_mass_minimum(self):
        with pytest.raises(ValueError, match="20.0"):
            _check_model_compatibility(1.2, self._metadata(0.125))

    def test_compatible_with_long_duration_model(self):
        # chirp_mass=1.2 → 128s required; model is also 128s → OK
        _check_model_compatibility(1.2, self._metadata(1.0 / 128))

    def test_compatible_multibanded_domain(self):
        # MFD model with base delta_f=0.25 → 4s; chirp_mass=28.3 → 4s → OK
        _check_model_compatibility(28.3, self._mfd_metadata(0.25))

    def test_incompatible_multibanded_domain_bns_raises(self):
        # MFD model is 4s; chirp_mass=1.2 → 128s required → error
        with pytest.raises(ValueError, match="4s"):
            _check_model_compatibility(1.2, self._mfd_metadata(0.25))


# ---------------------------------------------------------------------------
# _extract_prior_from_metadata / _extract_prior_from_model
# ---------------------------------------------------------------------------


class TestExtractPriorFromMetadata:
    def test_returns_dict(self):
        result = _extract_prior_from_metadata(MOCK_MODEL_METADATA)
        assert isinstance(result, dict)

    def test_excludes_scalar_values_from_intrinsic(self):
        """Fixed scalar entries like luminosity_distance=100.0 must not survive."""
        result = _extract_prior_from_metadata(MOCK_MODEL_METADATA)
        # Both keys appear in the result, but as strings (from extrinsic_prior)
        assert isinstance(result["luminosity_distance"], str)
        assert isinstance(result["geocent_time"], str)

    def test_extrinsic_overrides_intrinsic(self):
        """The extrinsic prior's luminosity_distance must override the scalar."""
        result = _extract_prior_from_metadata(MOCK_MODEL_METADATA)
        assert result["luminosity_distance"] == (
            "bilby.core.prior.Uniform(minimum=100.0, maximum=1000.0)"
        )

    def test_string_intrinsic_params_present(self):
        result = _extract_prior_from_metadata(MOCK_MODEL_METADATA)
        for param in ("chirp_mass", "mass_ratio", "a_1", "a_2", "phase"):
            assert param in result

    def test_extrinsic_params_present(self):
        result = _extract_prior_from_metadata(MOCK_MODEL_METADATA)
        for param in ("dec", "ra", "psi", "geocent_time", "luminosity_distance"):
            assert param in result

    def test_default_strings_preserved(self):
        """'default' strings should be passed through for dingo_pipe to resolve."""
        result = _extract_prior_from_metadata(MOCK_MODEL_METADATA)
        assert result["dec"] == "default"
        assert result["tilt_1"] == "default"


class TestExtractPriorFromModel:
    def test_calls_torch_load(self, monkeypatch, mock_model):
        called = {}

        def fake_load(path, **kwargs):
            called["path"] = path
            return {"metadata": MOCK_MODEL_METADATA}

        monkeypatch.setattr("torch.load", fake_load)
        _extract_prior_from_model(mock_model)
        assert called["path"] == mock_model

    def test_returns_correct_prior(self, monkeypatch, mock_model):
        monkeypatch.setattr(
            "torch.load", lambda *a, **kw: {"metadata": MOCK_MODEL_METADATA}
        )
        result = _extract_prior_from_model(mock_model)
        assert result["chirp_mass"].startswith(
            "bilby.gw.prior.UniformInComponentsChirpMass"
        )


# ---------------------------------------------------------------------------
# _write_config_file
# ---------------------------------------------------------------------------


class TestWriteConfigFile:
    def test_keys_use_hyphens(self, tmp_path):
        fname = str(tmp_path / "test.ini")
        _write_config_file({"trigger_time": 1234567890.0, "num_samples": 50000}, fname)
        content = Path(fname).read_text()
        assert "trigger-time=1234567890.0" in content
        assert "num-samples=50000" in content

    def test_dict_value_serialized_as_json(self, tmp_path):
        fname = str(tmp_path / "test.ini")
        _write_config_file({"psd_dict": {"H1": "H1_psd.txt"}}, fname)
        content = Path(fname).read_text()
        assert "psd-dict=" in content
        assert "H1_psd.txt" in content

    def test_comment_written(self, tmp_path):
        fname = str(tmp_path / "test.ini")
        _write_config_file({}, fname, comment="test event")
        assert "# test event" in Path(fname).read_text()


# ---------------------------------------------------------------------------
# prepare_dingo_config
# ---------------------------------------------------------------------------


class TestPrepareDingoConfig:
    def test_creates_ini_file(self, tmp_path, mock_model, mock_metadata):
        fname = prepare_dingo_config(MOCK_CANDIDATE_BBH, "T123456a", str(tmp_path), mock_model)
        assert os.path.isfile(fname)
        assert fname.endswith("dingo_config.ini")

    def test_ini_has_required_dingo_keys(self, tmp_path, mock_model, mock_metadata):
        fname = prepare_dingo_config(
            MOCK_CANDIDATE_BBH, "T123456a", str(tmp_path), mock_model,
            device="cpu", num_samples=1000,
        )
        content = Path(fname).read_text()
        assert "model=" in content
        assert "device=cpu" in content
        assert "num-samples=1000" in content
        assert "importance-sample=True" in content
        assert "recover-log-prob=True" in content

    def test_ini_has_data_generation_keys(self, tmp_path, mock_model, mock_metadata):
        fname = prepare_dingo_config(MOCK_CANDIDATE_BBH, "T123456a", str(tmp_path), mock_model)
        content = Path(fname).read_text()
        assert "trigger-time=" in content
        assert "detectors=" in content
        assert "label=T123456a" in content

    def test_model_owned_settings_not_in_config(self, tmp_path, mock_model, mock_metadata):
        """duration, reference_frequency, deltaT come from the model, not the INI."""
        fname = prepare_dingo_config(MOCK_CANDIDATE_BBH, "T123456a", str(tmp_path), mock_model)
        content = Path(fname).read_text()
        assert "duration=" not in content
        assert "reference-frequency=" not in content
        assert "deltaT=" not in content

    def test_no_prior_dict_in_config(self, tmp_path, mock_model, mock_metadata):
        fname = prepare_dingo_config(MOCK_CANDIDATE_BBH, "T123456a", str(tmp_path), mock_model)
        content = Path(fname).read_text()
        assert "prior-dict=" not in content
        assert "prior-file=" not in content

    def test_incompatible_bns_event_raises(self, tmp_path, mock_model, mock_metadata):
        """BNS event against an 8s BBH model must raise ValueError before writing."""
        with pytest.raises(ValueError, match="BNS or NSBH"):
            prepare_dingo_config(
                MOCK_CANDIDATE_BNS, "T654321b", str(tmp_path / "bns"), mock_model
            )
        # No config file should have been written
        assert not os.path.exists(str(tmp_path / "bns" / "dingo_config.ini"))

    def test_no_calibration_keys_when_off_cluster(self, tmp_path, mock_model, mock_metadata):
        fname = prepare_dingo_config(MOCK_CANDIDATE_BBH, "T123456a", str(tmp_path), mock_model)
        content = Path(fname).read_text()
        assert "calibration-model" not in content
        assert "spline-calibration-envelope-dict" not in content

    def test_calibration_keys_when_available(self, tmp_path, mock_model, mock_metadata, monkeypatch):
        import dingo.pipe.gracedb as gracedb_mod
        monkeypatch.setattr(
            gracedb_mod,
            "calibration_dict_lookup",
            lambda *a, **kw: (
                "CubicSpline",
                {"H1": "/path/H1.txt", "L1": "/path/L1.txt"},
            ),
        )
        fname = prepare_dingo_config(MOCK_CANDIDATE_BBH, "T123456a", str(tmp_path), mock_model)
        content = Path(fname).read_text()
        assert "calibration-model=CubicSpline" in content
        assert "spline-calibration-envelope-dict" in content
        assert "spline-calibration-nodes=10" in content

    def test_no_importance_sampling(self, tmp_path, mock_model, mock_metadata):
        fname = prepare_dingo_config(
            MOCK_CANDIDATE_BBH, "T123456a", str(tmp_path), mock_model,
            importance_sample=False,
        )
        assert "importance-sample=False" in Path(fname).read_text()

    def test_settings_override_accounting(self, tmp_path, mock_model, mock_metadata):
        fname = prepare_dingo_config(
            MOCK_CANDIDATE_BBH, "T123456a", str(tmp_path), mock_model,
            settings={"accounting": "ligo.prod.o4.cbc.pe.dingo"},
        )
        assert "accounting=ligo.prod.o4.cbc.pe.dingo" in Path(fname).read_text()

    def test_channel_dict_written(self, tmp_path, mock_model, mock_metadata):
        ch = {"H1": "GDS-CALIB_STRAIN_CLEAN", "L1": "GDS-CALIB_STRAIN_CLEAN"}
        fname = prepare_dingo_config(
            MOCK_CANDIDATE_BBH, "T123456a", str(tmp_path), mock_model, channel_dict=ch
        )
        assert "channel-dict=" in Path(fname).read_text()

    def test_batch_size_written_when_provided(self, tmp_path, mock_model, mock_metadata):
        fname = prepare_dingo_config(
            MOCK_CANDIDATE_BBH, "T123456a", str(tmp_path), mock_model, batch_size=10000
        )
        assert "batch-size=10000" in Path(fname).read_text()


# ---------------------------------------------------------------------------
# create_parser
# ---------------------------------------------------------------------------


class TestCreateParser:
    def test_requires_model(self):
        with pytest.raises(SystemExit):
            create_parser().parse_args(["--json", "event.json"])

    def test_requires_event_source(self):
        with pytest.raises(SystemExit):
            create_parser().parse_args(["--model", "model.pt"])

    def test_gracedb_and_json_mutually_exclusive(self):
        with pytest.raises(SystemExit):
            create_parser().parse_args(
                ["--gracedb", "S123456a", "--json", "x.json", "--model", "m.pt"]
            )

    def test_defaults(self):
        args = create_parser().parse_args(["--json", "x.json", "--model", "m.pt"])
        assert args.device == "cuda"
        assert args.num_samples == 50000
        assert args.importance_sample is True
        assert args.output == "full"
        assert args.channel_dict == "online"
        assert args.psd_cut == 0.95

    def test_no_importance_sampling_flag(self):
        args = create_parser().parse_args(
            ["--json", "x.json", "--model", "m.pt", "--no-importance-sampling"]
        )
        assert args.importance_sample is False

    def test_custom_device(self):
        args = create_parser().parse_args(
            ["--json", "x.json", "--model", "m.pt", "--device", "cpu"]
        )
        assert args.device == "cpu"

    def test_json_and_model_parsed(self):
        args = create_parser().parse_args(["--json", "event.json", "--model", "model.pt"])
        assert args.json == "event.json"
        assert args.model == "model.pt"
        assert args.gracedb is None

    def test_gracedb_parsed(self):
        args = create_parser().parse_args(["--gracedb", "S230914ax", "--model", "m.pt"])
        assert args.gracedb == "S230914ax"
        assert args.json is None

    def test_no_skymap_args(self):
        """Skymap args were removed since the prior now comes from the model."""
        option_strings = [
            a.option_strings[0]
            for a in create_parser()._actions
            if a.option_strings
        ]
        assert "--skymap-file" not in option_strings
        assert "--disable-skymap-download" not in option_strings


# ---------------------------------------------------------------------------
# main() end-to-end
# ---------------------------------------------------------------------------


class TestMain:
    def test_generates_config_from_json(self, tmp_path, mock_model, candidate_json, mock_metadata):
        from dingo.pipe.gracedb import main

        args = create_parser().parse_args(
            [
                "--json", candidate_json,
                "--model", mock_model,
                "--outdir", str(tmp_path),
                "--output", "ini",
                "--device", "cpu",
            ]
        )
        main(args)
        assert os.path.isfile(str(tmp_path / "dingo_config.ini"))

    def test_generated_config_has_correct_label(self, tmp_path, mock_model, candidate_json, mock_metadata):
        from dingo.pipe.gracedb import main

        args = create_parser().parse_args(
            [
                "--json", candidate_json,
                "--model", mock_model,
                "--outdir", str(tmp_path),
                "--output", "ini",
                "--device", "cpu",
            ]
        )
        main(args)

        kv = {}
        for line in (tmp_path / "dingo_config.ini").read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, _, v = line.partition("=")
                kv[k.strip()] = v.strip()

        assert kv["label"] == "T123456a"
        assert kv["model"] == mock_model

    def test_settings_json_applied(self, tmp_path, mock_model, candidate_json, mock_metadata):
        from dingo.pipe.gracedb import main

        settings_file = tmp_path / "overrides.json"
        settings_file.write_text(json.dumps({"accounting": "ligo.prod.o4.cbc.pe.dingo"}))

        args = create_parser().parse_args(
            [
                "--json", candidate_json,
                "--model", mock_model,
                "--outdir", str(tmp_path),
                "--output", "ini",
                "--settings", str(settings_file),
                "--device", "cpu",
            ]
        )
        main(args)

        content = (tmp_path / "dingo_config.ini").read_text()
        assert "accounting=ligo.prod.o4.cbc.pe.dingo" in content


# ---------------------------------------------------------------------------
# PSD extraction and psd_cut frequency capping
# ---------------------------------------------------------------------------


def _patch_psd_extraction(monkeypatch, psd_max_freq):
    """Replace extract_psds_from_xml with a fake; returns a dict recording calls."""
    import dingo.pipe.gracedb as gracedb_mod

    calls = {}

    def fake_extract(coinc_file, ifos, outdir):
        calls["coinc_file"] = coinc_file
        return (
            {ifo: f"{outdir}/{ifo}_psd.txt" for ifo in ifos},
            psd_max_freq,
        )

    monkeypatch.setattr(gracedb_mod, "extract_psds_from_xml", fake_extract)
    return calls


class TestPsdHandling:
    def test_psd_dict_written_when_coinc_present(
        self, tmp_path, mock_model, mock_metadata, monkeypatch
    ):
        calls = _patch_psd_extraction(monkeypatch, 4096.0)
        candidate = {**MOCK_CANDIDATE_BBH, "coinc_file": "coinc.xml"}
        fname = prepare_dingo_config(candidate, "T123456a", str(tmp_path), mock_model)
        assert "psd-dict=" in Path(fname).read_text()
        assert calls["coinc_file"] == "coinc.xml"

    def test_psd_cut_caps_maximum_frequency(
        self, tmp_path, mock_model, mock_metadata, monkeypatch
    ):
        """A PSD reaching only 800 Hz caps maximum_frequency at 0.95 * 800 = 760 Hz."""
        _patch_psd_extraction(monkeypatch, 800.0)
        candidate = {**MOCK_CANDIDATE_BBH, "coinc_file": "coinc.xml"}
        fname = prepare_dingo_config(candidate, "T123456a", str(tmp_path), mock_model)
        assert "maximum-frequency=760.0" in Path(fname).read_text()

    def test_wideband_psd_keeps_default_maximum_frequency(
        self, tmp_path, mock_model, mock_metadata, monkeypatch
    ):
        """A full-bandwidth PSD (0.95 * 4096 > 1024) leaves the 1024 Hz default."""
        _patch_psd_extraction(monkeypatch, 4096.0)
        candidate = {**MOCK_CANDIDATE_BBH, "coinc_file": "coinc.xml"}
        fname = prepare_dingo_config(candidate, "T123456a", str(tmp_path), mock_model)
        assert "maximum-frequency=1024.0" in Path(fname).read_text()

    def test_explicit_psd_cut(self, tmp_path, mock_model, mock_metadata, monkeypatch):
        _patch_psd_extraction(monkeypatch, 800.0)
        candidate = {**MOCK_CANDIDATE_BBH, "coinc_file": "coinc.xml"}
        fname = prepare_dingo_config(
            candidate, "T123456a", str(tmp_path), mock_model, psd_cut=0.5
        )
        assert "maximum-frequency=400.0" in Path(fname).read_text()


# ---------------------------------------------------------------------------
# Calibration lookup fallback
# ---------------------------------------------------------------------------


class TestCalibrationFallback:
    def test_missing_archive_does_not_crash(
        self, tmp_path, mock_model, mock_metadata, monkeypatch
    ):
        """Off-cluster (or in a container without /home/cal) the lookup raises
        FileNotFoundError; config generation must proceed without calibration."""
        import dingo.pipe.gracedb as gracedb_mod

        def raise_missing(*args, **kwargs):
            raise FileNotFoundError("/home/cal/public_html/archive/H1/uncertainty/v0")

        monkeypatch.setattr(gracedb_mod, "calibration_dict_lookup", raise_missing)
        fname = prepare_dingo_config(MOCK_CANDIDATE_BBH, "T123456a", str(tmp_path), mock_model)
        content = Path(fname).read_text()
        assert "calibration-model" not in content
        assert "spline-calibration-envelope-dict" not in content


# ---------------------------------------------------------------------------
# main() output modes and --psd-file override
# ---------------------------------------------------------------------------


class TestMainOutputModes:
    def _run_main(self, tmp_path, mock_model, candidate_json, output, monkeypatch):
        import dingo.pipe.gracedb as gracedb_mod
        from dingo.pipe.gracedb import main

        recorded = {}

        def fake_run(cmd, check=True, **kwargs):
            recorded["cmd"] = cmd
            recorded["check"] = check

        monkeypatch.setattr(gracedb_mod.subprocess, "run", fake_run)
        args = create_parser().parse_args(
            [
                "--json", candidate_json,
                "--model", mock_model,
                "--outdir", str(tmp_path),
                "--output", output,
                "--device", "cpu",
            ]
        )
        main(args)
        return recorded

    def test_output_ini_does_not_invoke_dingo_pipe(
        self, tmp_path, mock_model, candidate_json, mock_metadata, monkeypatch
    ):
        recorded = self._run_main(tmp_path, mock_model, candidate_json, "ini", monkeypatch)
        assert recorded == {}

    def test_output_full_invokes_dingo_pipe(
        self, tmp_path, mock_model, candidate_json, mock_metadata, monkeypatch
    ):
        recorded = self._run_main(tmp_path, mock_model, candidate_json, "full", monkeypatch)
        assert recorded["cmd"] == ["dingo_pipe", str(tmp_path / "dingo_config.ini")]
        assert recorded["check"] is True

    def test_output_full_local_appends_local_flag(
        self, tmp_path, mock_model, candidate_json, mock_metadata, monkeypatch
    ):
        recorded = self._run_main(
            tmp_path, mock_model, candidate_json, "full-local", monkeypatch
        )
        assert recorded["cmd"][-1] == "--local"

    def test_output_full_submit_appends_submit_flag(
        self, tmp_path, mock_model, candidate_json, mock_metadata, monkeypatch
    ):
        recorded = self._run_main(
            tmp_path, mock_model, candidate_json, "full-submit", monkeypatch
        )
        assert recorded["cmd"][-1] == "--submit"


class TestMainPsdFileOverride:
    def test_psd_file_overrides_coinc_file(
        self, tmp_path, mock_model, candidate_json, mock_metadata, monkeypatch
    ):
        from dingo.pipe.gracedb import main

        calls = _patch_psd_extraction(monkeypatch, 4096.0)
        psd_file = tmp_path / "coinc_override.xml"
        psd_file.write_text("<xml/>")
        args = create_parser().parse_args(
            [
                "--json", candidate_json,
                "--model", mock_model,
                "--outdir", str(tmp_path),
                "--output", "ini",
                "--psd-file", str(psd_file),
                "--device", "cpu",
            ]
        )
        main(args)
        assert calls["coinc_file"] == str(psd_file)
        assert "psd-dict=" in (tmp_path / "dingo_config.ini").read_text()

    def test_nonexistent_psd_file_raises(
        self, tmp_path, mock_model, candidate_json, mock_metadata, monkeypatch
    ):
        """A typo'd --psd-file must fail early, not silently fall back to a
        different (or no) PSD source."""
        from dingo.pipe.gracedb import main

        calls = _patch_psd_extraction(monkeypatch, 4096.0)
        args = create_parser().parse_args(
            [
                "--json", candidate_json,
                "--model", mock_model,
                "--outdir", str(tmp_path),
                "--output", "ini",
                "--psd-file", str(tmp_path / "does_not_exist.xml"),
                "--device", "cpu",
            ]
        )
        with pytest.raises(ValueError, match="does not exist"):
            main(args)
        assert calls == {}


# ---------------------------------------------------------------------------
# GNPE init model
# ---------------------------------------------------------------------------


class TestIsGnpeModel:
    def test_plain_model_is_not_gnpe(self):
        assert not _is_gnpe_model(MOCK_MODEL_METADATA)

    def test_gnpe_time_shifts_detected(self):
        assert _is_gnpe_model(MOCK_GNPE_MODEL_METADATA)

    def test_gnpe_phase_detected(self):
        metadata = copy.deepcopy(MOCK_MODEL_METADATA)
        metadata["train_settings"]["data"]["gnpe_phase"] = {"kernel": "..."}
        assert _is_gnpe_model(metadata)


class TestFindInitModel:
    def test_sibling_init_found(self, tmp_path):
        (tmp_path / "network.pt").write_bytes(b"main")
        (tmp_path / "network_init.pt").write_bytes(b"init")
        assert _find_init_model(str(tmp_path / "network.pt")) == str(
            tmp_path / "network_init.pt"
        )

    def test_main_naming_convention(self, tmp_path):
        """model_A_main.pt pairs with model_A_init.pt (O4c convention)."""
        (tmp_path / "model_A_main.pt").write_bytes(b"main")
        (tmp_path / "model_A_init.pt").write_bytes(b"init")
        assert _find_init_model(str(tmp_path / "model_A_main.pt")) == str(
            tmp_path / "model_A_init.pt"
        )

    def test_no_init_returns_none(self, tmp_path):
        (tmp_path / "network.pt").write_bytes(b"main")
        assert _find_init_model(str(tmp_path / "network.pt")) is None

    def test_main_convention_takes_precedence(self, tmp_path):
        """For X_main.pt, the documented X_init.pt wins over X_main_init.pt."""
        (tmp_path / "model_A_main.pt").write_bytes(b"main")
        (tmp_path / "model_A_init.pt").write_bytes(b"init")
        (tmp_path / "model_A_main_init.pt").write_bytes(b"stale init")
        assert _find_init_model(str(tmp_path / "model_A_main.pt")) == str(
            tmp_path / "model_A_init.pt"
        )


class TestGnpeInitModel:
    def test_explicit_model_init_written(self, tmp_path, mock_model, mock_gnpe_metadata):
        init_path = tmp_path / "custom_init.pt"
        init_path.write_bytes(b"fake init model")
        fname = prepare_dingo_config(
            MOCK_CANDIDATE_BBH, "T123456a", str(tmp_path), mock_model,
            model_init=str(init_path),
        )
        assert f"model-init={init_path}" in Path(fname).read_text()

    def test_explicit_model_init_missing_file_raises(
        self, tmp_path, mock_model, mock_gnpe_metadata
    ):
        with pytest.raises(ValueError, match="does not exist"):
            prepare_dingo_config(
                MOCK_CANDIDATE_BBH, "T123456a", str(tmp_path), mock_model,
                model_init=str(tmp_path / "no_such_init.pt"),
            )

    def test_model_init_with_non_gnpe_model_raises(
        self, tmp_path, mock_model, mock_metadata
    ):
        """A non-GNPE model with --model-init would make sampling use the
        wrong sampler; refuse at config time."""
        init_path = tmp_path / "custom_init.pt"
        init_path.write_bytes(b"fake init model")
        with pytest.raises(ValueError, match="not a GNPE"):
            prepare_dingo_config(
                MOCK_CANDIDATE_BBH, "T123456a", str(tmp_path), mock_model,
                model_init=str(init_path),
            )

    def test_gnpe_autodetects_sibling_init(self, tmp_path, mock_model, mock_gnpe_metadata):
        # mock_model is tmp_path/model.pt; create the conventional sibling
        init_path = tmp_path / "model_init.pt"
        init_path.write_bytes(b"fake init model")
        fname = prepare_dingo_config(
            MOCK_CANDIDATE_BBH, "T123456a", str(tmp_path), mock_model
        )
        assert f"model-init={init_path}" in Path(fname).read_text()

    def test_gnpe_without_init_raises(self, tmp_path, mock_model, mock_gnpe_metadata):
        with pytest.raises(ValueError, match="init model"):
            prepare_dingo_config(
                MOCK_CANDIDATE_BBH, "T123456a", str(tmp_path / "out"), mock_model
            )
        assert not os.path.exists(str(tmp_path / "out" / "dingo_config.ini"))

    def test_non_gnpe_has_no_model_init(self, tmp_path, mock_model, mock_metadata):
        # Even with a sibling init file present, a non-GNPE model gets none.
        (tmp_path / "model_init.pt").write_bytes(b"fake init model")
        fname = prepare_dingo_config(
            MOCK_CANDIDATE_BBH, "T123456a", str(tmp_path), mock_model
        )
        assert "model-init=" not in Path(fname).read_text()

    def test_parser_accepts_model_init(self, mock_model):
        args = create_parser().parse_args(
            ["--json", "e.json", "--model", mock_model, "--model-init", "init.pt"]
        )
        assert args.model_init == "init.pt"

    def test_parser_model_init_defaults_none(self, mock_model):
        args = create_parser().parse_args(["--json", "e.json", "--model", mock_model])
        assert args.model_init is None

    def test_main_passes_model_init_through(
        self, tmp_path, mock_model, candidate_json, mock_gnpe_metadata
    ):
        from dingo.pipe.gracedb import main

        init_path = tmp_path / "custom_init.pt"
        init_path.write_bytes(b"fake init model")
        args = create_parser().parse_args(
            [
                "--json", candidate_json,
                "--model", mock_model,
                "--model-init", str(init_path),
                "--outdir", str(tmp_path),
                "--output", "ini",
                "--device", "cpu",
            ]
        )
        main(args)
        content = (tmp_path / "dingo_config.ini").read_text()
        assert f"model-init={init_path}" in content
