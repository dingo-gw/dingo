"""Integration tests to verify asimov manage build compatibility."""

import shutil
import tempfile
from importlib.resources import path as resources_path
from unittest.mock import MagicMock

import pytest

from dingo.asimov.asimov import Dingo

try:
    from asimov.pipeline import Pipeline
    from asimov.pipelines import known_pipelines

    ASIMOV_AVAILABLE = True
except ModuleNotFoundError:
    ASIMOV_AVAILABLE = False


@pytest.mark.skipif(not ASIMOV_AVAILABLE, reason="asimov not installed")
class TestAsimovManageBuild:
    """Tests to verify asimov manage build will work with Dingo pipeline."""

    @staticmethod
    def _make_production(ini_path, rundir=None):
        """Create a minimal mock production."""
        production = MagicMock()
        production.pipeline = "dingo"
        production.name = "test_production"
        production.meta = {}
        production.category = None
        production.rundir = rundir or "/tmp/test"
        production.event = MagicMock()
        production.event.name = "test_event"
        production.status = "wait"
        production.job_id = None
        production.event.repository = MagicMock()
        production.event.repository.find_prods = MagicMock(return_value=[ini_path])
        return production

    def test_dingo_in_known_pipelines(self):
        """Test that Dingo is registered in asimov's known_pipelines."""
        assert "dingo" in known_pipelines

    def test_dingo_pipeline_from_ini(self):
        """Test that a Dingo pipeline can be created using ini configuration."""
        with resources_path("dingo.asimov", "dingo.ini") as ini_path:
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_ini = shutil.copy(ini_path, tmpdir)
                production = self._make_production(temp_ini, tmpdir)
                pipeline = Dingo(production)

                assert isinstance(pipeline, Pipeline)
                assert pipeline.name == "dingo"
