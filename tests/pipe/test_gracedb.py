"""Tests for dingo.pipe.gracedb (dingo_pipe_gracedb).

All tests are self-contained: no GraceDB network access, no real dingo model,
and no CIT cluster (calibration lookup fails gracefully off-cluster).
"""

import json
import os

import pytest

from dingo.pipe.gracedb import (
    _extract_prior_from_model,
    _get_analysis_duration,
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

# BNS event: low chirp mass → long duration
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

# Minimal model metadata matching the structure of a real dingo checkpoint.
MOCK_MODEL_METADATA = {
    "dataset_settings": {
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
    # From intrinsic (strings only)
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
    # From extrinsic (overrides intrinsic for shared parameters)
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
def mock_prior(monkeypatch):
    """Patch _extract_prior_from_model to return a canned prior dict."""
    import dingo.pipe.gracedb as gracedb_mod
    monkeypatch.setattr(gracedb_mod, "_extract_prior_from_model", lambda _: MOCK_PRIOR_DICT)


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
# _extract_prior_from_model
# ---------------------------------------------------------------------------


class TestExtractPriorFromModel:
    def _mock_load(self, path, **kwargs):
        return {"metadata": MOCK_MODEL_METADATA}

    def test_returns_dict(self, monkeypatch, mock_model):
        monkeypatch.setattr("torch.load", self._mock_load)
        result = _extract_prior_from_model(mock_model)
        assert isinstance(result, dict)

    def test_excludes_scalar_values(self, monkeypatch, mock_model):
        """Fixed scalar values (e.g. luminosity_distance=100.0) must not appear."""
        monkeypatch.setattr("torch.load", self._mock_load)
        result = _extract_prior_from_model(mock_model)
        # From intrinsic_prior, luminosity_distance=100.0 and geocent_time=0.0 are scalars
        # but extrinsic_prior has string versions, so they should appear as strings
        assert isinstance(result.get("luminosity_distance"), str)
        assert isinstance(result.get("geocent_time"), str)

    def test_extrinsic_overrides_intrinsic(self, monkeypatch, mock_model):
        """Extrinsic prior's luminosity_distance must override the intrinsic scalar."""
        monkeypatch.setattr("torch.load", self._mock_load)
        result = _extract_prior_from_model(mock_model)
        assert result["luminosity_distance"] == (
            "bilby.core.prior.Uniform(minimum=100.0, maximum=1000.0)"
        )

    def test_string_intrinsic_params_present(self, monkeypatch, mock_model):
        """All string-valued intrinsic prior entries should be included."""
        monkeypatch.setattr("torch.load", self._mock_load)
        result = _extract_prior_from_model(mock_model)
        for param in ("chirp_mass", "mass_ratio", "a_1", "a_2", "phase"):
            assert param in result

    def test_extrinsic_params_present(self, monkeypatch, mock_model):
        """All extrinsic prior entries should be included."""
        monkeypatch.setattr("torch.load", self._mock_load)
        result = _extract_prior_from_model(mock_model)
        for param in ("dec", "ra", "psi", "geocent_time", "luminosity_distance"):
            assert param in result

    def test_default_strings_preserved(self, monkeypatch, mock_model):
        """'default' strings should be kept as-is for dingo_pipe to resolve."""
        monkeypatch.setattr("torch.load", self._mock_load)
        result = _extract_prior_from_model(mock_model)
        assert result["dec"] == "default"
        assert result["tilt_1"] == "default"


# ---------------------------------------------------------------------------
# _write_config_file
# ---------------------------------------------------------------------------


class TestWriteConfigFile:
    def test_keys_use_hyphens(self, tmp_path):
        fname = str(tmp_path / "test.ini")
        _write_config_file({"trigger_time": 1234567890.0, "num_samples": 50000}, fname)
        content = open(fname).read()
        assert "trigger-time=1234567890.0" in content
        assert "num-samples=50000" in content

    def test_dict_value_serialized_as_json(self, tmp_path):
        fname = str(tmp_path / "test.ini")
        _write_config_file({"psd_dict": {"H1": "H1_psd.txt"}}, fname)
        content = open(fname).read()
        assert "psd-dict=" in content
        assert "H1_psd.txt" in content

    def test_comment_written(self, tmp_path):
        fname = str(tmp_path / "test.ini")
        _write_config_file({}, fname, comment="test event")
        assert "# test event" in open(fname).read()


# ---------------------------------------------------------------------------
# prepare_dingo_config
# ---------------------------------------------------------------------------


class TestPrepareDingoConfig:
    def test_creates_ini_file(self, tmp_path, mock_model, mock_prior):
        fname = prepare_dingo_config(MOCK_CANDIDATE_BBH, "T123456a", str(tmp_path), mock_model)
        assert os.path.isfile(fname)
        assert fname.endswith("dingo_config.ini")

    def test_ini_has_required_dingo_keys(self, tmp_path, mock_model, mock_prior):
        fname = prepare_dingo_config(
            MOCK_CANDIDATE_BBH, "T123456a", str(tmp_path), mock_model,
            device="cpu", num_samples=1000,
        )
        content = open(fname).read()
        assert "model=" in content
        assert "device=cpu" in content
        assert "num-samples=1000" in content
        assert "importance-sample=True" in content
        assert "recover-log-prob=True" in content

    def test_ini_has_data_generation_keys(self, tmp_path, mock_model, mock_prior):
        fname = prepare_dingo_config(MOCK_CANDIDATE_BBH, "T123456a", str(tmp_path), mock_model)
        content = open(fname).read()
        assert "trigger-time=" in content
        assert "detectors=" in content
        assert "duration=" in content
        assert "label=T123456a" in content

    def test_prior_dict_from_model(self, tmp_path, mock_model, mock_prior):
        """Config must use prior-dict (not prior-file), taken from the model."""
        fname = prepare_dingo_config(MOCK_CANDIDATE_BBH, "T123456a", str(tmp_path), mock_model)
        content = open(fname).read()
        assert "prior-dict=" in content
        assert "prior-file=" not in content
        # The model's chirp mass prior range must be in the config
        assert "20.0" in content
        assert "120.0" in content

    def test_no_calibration_keys_when_off_cluster(self, tmp_path, mock_model, mock_prior):
        fname = prepare_dingo_config(MOCK_CANDIDATE_BBH, "T123456a", str(tmp_path), mock_model)
        content = open(fname).read()
        assert "calibration-model" not in content
        assert "spline-calibration-envelope-dict" not in content

    def test_calibration_keys_when_available(self, tmp_path, mock_model, mock_prior, monkeypatch):
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
        content = open(fname).read()
        assert "calibration-model=CubicSpline" in content
        assert "spline-calibration-envelope-dict" in content
        assert "spline-calibration-nodes=10" in content

    def test_no_importance_sampling(self, tmp_path, mock_model, mock_prior):
        fname = prepare_dingo_config(
            MOCK_CANDIDATE_BBH, "T123456a", str(tmp_path), mock_model,
            importance_sample=False,
        )
        assert "importance-sample=False" in open(fname).read()

    def test_settings_override_accounting(self, tmp_path, mock_model, mock_prior):
        fname = prepare_dingo_config(
            MOCK_CANDIDATE_BBH, "T123456a", str(tmp_path), mock_model,
            settings={"accounting": "ligo.prod.o4.cbc.pe.dingo"},
        )
        assert "accounting=ligo.prod.o4.cbc.pe.dingo" in open(fname).read()

    def test_bns_duration_longer_than_bbh(self, tmp_path, mock_model, mock_prior):
        bbh_dir = tmp_path / "bbh"
        bns_dir = tmp_path / "bns"
        bbh_dir.mkdir()
        bns_dir.mkdir()

        fname_bbh = prepare_dingo_config(
            MOCK_CANDIDATE_BBH, "T123456a", str(bbh_dir), mock_model
        )
        fname_bns = prepare_dingo_config(
            MOCK_CANDIDATE_BNS, "T654321b", str(bns_dir), mock_model
        )

        def get_duration(path):
            for line in open(path):
                if line.startswith("duration="):
                    return int(line.split("=")[1].strip())
            return None

        assert get_duration(fname_bns) > get_duration(fname_bbh)

    def test_channel_dict_written(self, tmp_path, mock_model, mock_prior):
        ch = {"H1": "GDS-CALIB_STRAIN_CLEAN", "L1": "GDS-CALIB_STRAIN_CLEAN"}
        fname = prepare_dingo_config(
            MOCK_CANDIDATE_BBH, "T123456a", str(tmp_path), mock_model, channel_dict=ch
        )
        assert "channel-dict=" in open(fname).read()

    def test_batch_size_written_when_provided(self, tmp_path, mock_model, mock_prior):
        fname = prepare_dingo_config(
            MOCK_CANDIDATE_BBH, "T123456a", str(tmp_path), mock_model, batch_size=10000
        )
        assert "batch-size=10000" in open(fname).read()


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
        parser = create_parser()
        option_strings = [
            a.option_strings[0]
            for a in parser._actions
            if a.option_strings
        ]
        assert "--skymap-file" not in option_strings
        assert "--disable-skymap-download" not in option_strings


# ---------------------------------------------------------------------------
# main() end-to-end
# ---------------------------------------------------------------------------


class TestMain:
    def test_generates_config_from_json(self, tmp_path, mock_model, candidate_json, mock_prior):
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

    def test_generated_config_has_correct_label(self, tmp_path, mock_model, candidate_json, mock_prior):
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

    def test_settings_json_applied(self, tmp_path, mock_model, candidate_json, mock_prior):
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
