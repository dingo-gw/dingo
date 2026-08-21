import pytest

from dingo.gw.approximant import Approximant
from dingo.gw.imports import check_function_signature, import_entity


def test_import_entity() -> None:

    # UniformFrequencyDomain should be imported with success
    d, module, d_name = import_entity(
        "dingo.gw.domains.uniform_frequency_domain.UniformFrequencyDomain"
    )
    assert d_name == "UniformFrequencyDomain"
    assert module == "dingo.gw.domains.uniform_frequency_domain"
    assert d.__name__ == "UniformFrequencyDomain"

    # get_approximant should be imported with success
    f, module, f_name = import_entity("dingo.gw.approximant.get_approximant")
    assert f_name == "get_approximant"
    assert module == "dingo.gw.approximant"
    assert check_function_signature(f, [Approximant], int)

    # this import should fail
    with pytest.raises(ImportError):
        import_entity("package.module.function")

    # this should fail as well, except if an italian starts to program domains.
    with pytest.raises(ImportError):
        import_entity("dingo.gw.domains.RavioliDomain")


def test_check_function_signature() -> None:

    def fn(a: int, b: float) -> float:
        return a + b

    # correct signature for fn
    assert check_function_signature(fn, [int, float], float)

    # incorrect signature for fn
    assert not check_function_signature(fn, [int], str)

    # suitable transform function
    def transform(a: int) -> int:
        return a

    assert check_function_signature(transform, [int], int)

    # unsuitable transform: extra required param
    def not_transform(a: int, b: float) -> int:
        return a

    assert not check_function_signature(not_transform, [int], int)

    # suitable transform with optional kwarg
    def kwargs_transform(a: int, b: float = 0.0) -> int:
        return a

    assert check_function_signature(kwargs_transform, [int], int)
