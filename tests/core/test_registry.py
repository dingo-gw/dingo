"""Tests for dingo.core.registry: name resolution for pluggable components."""

import pytest

from dingo.core.registry import NEURAL_DISTRIBUTIONS, Registry


@pytest.fixture()
def registry():
    return Registry("test_components", entry_point_group="dingo.test_components")


def test_register_and_get(registry):
    @registry.register("my_component")
    class MyComponent:
        pass

    assert registry.get("my_component") is MyComponent
    assert "my_component" in registry
    assert registry.names() == ["my_component"]


def test_register_duplicate_name_raises(registry):
    @registry.register("taken")
    class ComponentA:
        pass

    with pytest.raises(ValueError, match="already registered"):

        @registry.register("taken")
        class ComponentB:
            pass


def test_register_same_component_twice_is_idempotent(registry):
    class MyComponent:
        pass

    registry.register("name")(MyComponent)
    registry.register("name")(MyComponent)
    assert registry.get("name") is MyComponent


def test_get_dotted_path(registry):
    from dingo.core.nn.enets import DenseResidualNet

    assert registry.get("dingo.core.nn.enets.DenseResidualNet") is DenseResidualNet


def test_get_file_path(registry, tmp_path):
    plugin = tmp_path / "my_plugin.py"
    plugin.write_text(
        "class MyNN:\n"
        "    marker = 'from-file'\n"
    )
    component = registry.get(f"{plugin}:MyNN")
    assert component.marker == "from-file"
    # Second lookup resolves from the cache to the same class object.
    assert registry.get(f"{plugin}:MyNN") is component


def test_get_unknown_name_raises_keyerror(registry):
    with pytest.raises(KeyError, match="test_components.*'nonexistent' not found"):
        registry.get("nonexistent")


def test_get_missing_file_raises_keyerror(registry, tmp_path):
    with pytest.raises(KeyError):
        registry.get(f"{tmp_path}/does_not_exist.py:MyNN")


def test_get_missing_attribute_raises_keyerror(registry, tmp_path):
    plugin = tmp_path / "my_plugin_2.py"
    plugin.write_text("class MyNN:\n    pass\n")
    with pytest.raises(KeyError):
        registry.get(f"{plugin}:WrongName")


def test_entry_point_resolution(registry, monkeypatch):
    class FakeEntryPoint:
        name = "installed_component"

        @staticmethod
        def load():
            return "the-component"

    def fake_entry_points(group):
        assert group == "dingo.test_components"
        return [FakeEntryPoint]

    monkeypatch.setattr("importlib.metadata.entry_points", fake_entry_points)
    assert registry.get("installed_component") == "the-component"


def test_builtin_distributions_are_registered():
    from dingo.core.posterior_models import (
        FlowMatchingPosteriorModel,
        NormalizingFlowPosteriorModel,
        ScoreDiffusionPosteriorModel,
    )

    assert NEURAL_DISTRIBUTIONS.get("normalizing_flow") is NormalizingFlowPosteriorModel
    assert NEURAL_DISTRIBUTIONS.get("flow_matching") is FlowMatchingPosteriorModel
    assert NEURAL_DISTRIBUTIONS.get("score_matching") is ScoreDiffusionPosteriorModel


def test_build_model_from_kwargs_with_file_path_plugin(tmp_path):
    """A NeuralDistribution defined in a user file (never pip-installed, not in the
    dingo source) can be selected as posterior_model_type via the file-path form."""
    from tests.core.test_build_model import model_settings

    plugin = tmp_path / "my_distribution.py"
    plugin.write_text(
        "from dingo.core.posterior_models import NormalizingFlowPosteriorModel\n"
        "\n"
        "class MyDistribution(NormalizingFlowPosteriorModel):\n"
        "    pass\n"
    )
    settings = model_settings("normalizing_flow")
    settings["train_settings"]["model"][
        "posterior_model_type"
    ] = f"{plugin}:MyDistribution"

    from dingo.core.posterior_models.build_model import build_model_from_kwargs

    pm = build_model_from_kwargs(settings=settings, device="cpu")
    assert type(pm).__name__ == "MyDistribution"


def test_backward_compatible_alias():
    """Other branches and downstream code import BasePosteriorModel; the alias must
    survive the NeuralDistribution rename."""
    from dingo.core.posterior_models import BasePosteriorModel, NeuralDistribution

    assert BasePosteriorModel is NeuralDistribution
