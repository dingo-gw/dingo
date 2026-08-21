import importlib
import inspect
import json
import tomllib
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    get_type_hints,
)

import yaml


def check_function_signature(
    fn: Callable, expected_param_types: List[Type], expected_return_type: Type
) -> bool:
    """
    Check if the function signature matches the expected signature.

    Parameters
    ----------
    fn : Callable
        The function to check.
    expected_param_types : List[type]
        The expected types of the function's parameters.
    expected_return_type : type
        The expected return type of the function.

    Returns
    -------
    bool
        True if the function signature matches, False otherwise.
    """

    sig = inspect.signature(fn)
    params = sig.parameters
    return_annotation = sig.return_annotation

    # Get type hints for the function
    type_hints = get_type_hints(fn)

    # Check return type
    if (
        return_annotation is not inspect.Signature.empty
        and return_annotation != expected_return_type
    ):
        return False

    # Check parameter types
    param_types = [type_hints.get(name, None) for name in params]

    # Check if all expected_param_types are in param_types
    for expected_type in expected_param_types:
        if expected_type not in param_types:
            return False

    # Check if all required parameters are covered
    for name, param in params.items():
        if (
            param.default is inspect.Parameter.empty
            and type_hints.get(name, None) not in expected_param_types
        ):
            return False

    return True


def import_entity(import_path: str) -> Tuple[Any, str, str]:
    """
    Given an import path as a string, returns a tuple containing the imported entity,
    the related module (as a string), and the entity name (as a string).

    Example of usage:

    ```
    domain_class, module_path, name = import_entity('dingo.gw.domains.UniformFrequencyDomain')

    # domain_class: the UniformFrequencyDomain class
    # module path: "dingo.gw.domains"
    # name: "UniformFrequencyDomain"
    ```

    Parameters
    ----------
    import_path : The import path as a string.

    Returns
    -------
    A tuple containing the imported entity, module path, and entity name.

    Raises
    ------
    ImportError
        If the import failed.
    """

    # Split the import path to get the module path and entity name
    *module_parts, entity_name = import_path.split(".")
    module_path = ".".join(module_parts)

    try:
        # Import the module dynamically
        module = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(f"Failed to import module '{module_path}': {e}")

    try:
        # Retrieve the entity from the module
        entity = getattr(module, entity_name)
    except AttributeError as e:
        raise ImportError(
            f"Could import module '{module_path}', but not '{entity_name}'. Error: {e}"
        )

    return entity, module_path, entity_name


def import_function(
    import_path_or_fn: Optional[Union[str, Callable]],
    expected_param_types: List[type],
    expected_return_type: type,
) -> Optional[Callable]:
    """
    Dynamically imports and validates a function based on its signature.

    This function serves two purposes:
    1. Imports a function from a specified import path
    2. Validates that the imported function matches the expected signature

    Parameters
    ----------
    import_path_or_fn :
        Either:
            - A string representing the import path (e.g., 'module.submodule.function_name')
            - An existing callable function to validate
            - None to return None immediately
    expected_param_types :
        List of expected parameter types that the function should accept
    expected_return_type :
        Expected return type of the function

    Returns
    -------
    The imported and validated function, or None if import_path_or_fn is None

    Raises
    ------
    ValueError
        If the function's signature does not match the expected parameters or return type
    ImportError
        If the module import fails or if the specified entity cannot be found
    """

    if import_path_or_fn is None:
        return None
    if not isinstance(import_path_or_fn, str):
        return import_path_or_fn
    fn, _, entity_name = import_entity(import_path_or_fn)
    proper_signature: bool = check_function_signature(
        fn, expected_param_types, expected_return_type
    )
    if not proper_signature:
        raise ValueError(
            f"imported function {entity_name} from import path {import_path_or_fn}, "
            f"but this function does not have the expected signature "
            f"(args: {expected_return_type}, return: {expected_return_type}"
        )
    return fn


def read_file(file_path: Union[Path, str]) -> Dict:
    if str(file_path).lower().endswith(".json"):
        with open(file_path, "r") as f:
            params = json.load(f)
    elif str(file_path).lower().endswith(".toml"):
        with open(file_path, "rb") as f:
            params = tomllib.load(f)
    elif str(file_path).lower().endswith((".yaml", ".yml")):
        with open(file_path, "r") as f:
            params = yaml.safe_load(f)
    else:
        raise ValueError(
            f"Unsupported file format: {file_path}. Only .json, .toml, .yaml, and .yml files are supported."
        )
    return params
