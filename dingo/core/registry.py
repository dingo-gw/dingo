"""
Registries for pluggable dingo components.

A Registry maps short, stable names (as stored in train settings and model metadata)
to component classes or builder functions. Components inside dingo register
themselves with the ``register`` decorator. Third-party components can be used
without editing dingo source; ``Registry.get`` resolves a name in this order:

1. a name registered via the decorator (dingo-internal components),
2. an installed entry point in the registry's entry-point group (pip-installed
   plugin packages),
3. a dotted import path, e.g. ``"my_package.nets.MyNN"``,
4. a file path with class name, e.g. ``"/path/to/my_net.py:MyNN"`` (un-packaged
   experiments).

Checkpoints should store the short name (form 1/2) where possible: it is stable
under dingo-internal refactors, unlike full import paths. See
hackathon/NN_Build_System_Design.md §4.2.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import importlib.util
import sys
from typing import Any, Callable, Dict, List

ARCHITECTURE_ENTRY_POINT_GROUP = "dingo.architectures"


class Registry:
    """
    A name -> component mapping with plugin resolution.

    Parameters
    ----------
    kind : str
        Human-readable name of the component kind (e.g. "neural_distributions").
        Used in error messages.
    entry_point_group : str
        Entry-point group searched for pip-installed plugins.
    """

    def __init__(
        self, kind: str, entry_point_group: str = ARCHITECTURE_ENTRY_POINT_GROUP
    ):
        self.kind = kind
        self.entry_point_group = entry_point_group
        self._components: Dict[str, Any] = {}

    def register(self, name: str) -> Callable:
        """
        Class/function decorator registering the component under ``name``.

        Raises ValueError if the name is already taken by a different component.
        """

        def decorator(component):
            existing = self._components.get(name)
            if existing is not None and existing is not component:
                raise ValueError(
                    f"{self.kind}: name '{name}' is already registered "
                    f"for {existing!r}."
                )
            self._components[name] = component
            return component

        return decorator

    def get(self, name: str) -> Any:
        """
        Resolve ``name`` to a component (see module docstring for the lookup order).

        Raises KeyError if the name cannot be resolved.
        """
        if name in self._components:
            return self._components[name]

        for resolve in (self._from_entry_points, self._from_dotted_path,
                        self._from_file_path):
            component = resolve(name)
            if component is not None:
                # Cache so repeated lookups are cheap and resolve consistently.
                self._components[name] = component
                return component

        raise KeyError(
            f"{self.kind}: '{name}' not found. Available names: {self.names()}. "
            f"For a plugin, is the providing package installed (entry-point group "
            f"'{self.entry_point_group}')? Alternatively use a dotted import path "
            f"('my_package.module.MyClass') or a file path "
            f"('/path/to/file.py:MyClass')."
        )

    def names(self) -> List[str]:
        """Names registered so far (excluding not-yet-loaded entry points)."""
        return sorted(self._components)

    def __contains__(self, name: str) -> bool:
        return name in self._components

    def _from_entry_points(self, name: str):
        for entry_point in importlib.metadata.entry_points(
            group=self.entry_point_group
        ):
            if entry_point.name == name:
                return entry_point.load()
        return None

    @staticmethod
    def _from_dotted_path(name: str):
        module_name, _, attribute = name.rpartition(".")
        if not module_name:
            return None
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            return None
        return getattr(module, attribute, None)

    @staticmethod
    def _from_file_path(name: str):
        path, separator, attribute = name.rpartition(":")
        if not separator or not path.endswith(".py"):
            return None
        module_name = f"_dingo_file_plugin_{abs(hash(path))}"
        if module_name in sys.modules:
            module = sys.modules[module_name]
        else:
            spec = importlib.util.spec_from_file_location(module_name, path)
            if spec is None or spec.loader is None:
                return None
            module = importlib.util.module_from_spec(spec)
            # Insert before exec so that e.g. dataclasses in the file resolve.
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except FileNotFoundError:
                del sys.modules[module_name]
                return None
        return getattr(module, attribute, None)


# The registries of the NN build system. Components register themselves where they
# are defined; nothing needs to be added here to introduce a new architecture.
NEURAL_DISTRIBUTIONS = Registry("neural_distributions")
EMBEDDING_NETS = Registry("embedding_networks")
CONTEXT_MERGERS = Registry("context_mergers")
