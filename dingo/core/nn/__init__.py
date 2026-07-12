"""Neural network building blocks and registered architectures.

Importing the architecture modules registers the built-in embedding networks and
context mergers with the registries (dingo.core.registry); importing any
dingo.core.nn submodule triggers this.
"""

import dingo.core.nn.enets  # noqa: F401
import dingo.core.nn.transformer  # noqa: F401
