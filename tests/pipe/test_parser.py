"""Tests for dingo_pipe parser options added for low-latency operation."""

import pytest

from dingo.pipe.parser import create_parser


@pytest.fixture
def empty_ini(tmp_path):
    p = tmp_path / "empty.ini"
    p.write_text("model=/fake/model.pt\n")
    return str(p)


class TestImportanceSamplingPool:
    def test_default_is_igwn_pool(self, empty_ini):
        args, _ = create_parser().parse_known_args([empty_ini])
        assert args.importance_sampling_pool == "igwn-pool"

    def test_local_pool_accepted(self, empty_ini):
        args, _ = create_parser().parse_known_args(
            [empty_ini, "--importance-sampling-pool", "local-pool"]
        )
        assert args.importance_sampling_pool == "local-pool"

    def test_invalid_pool_rejected(self, empty_ini):
        with pytest.raises(SystemExit):
            create_parser().parse_known_args(
                [empty_ini, "--importance-sampling-pool", "mars"]
            )
