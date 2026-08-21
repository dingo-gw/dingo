"""Small local `@deprecated` decorator.

Emits a `DeprecationWarning` when a function is called or a class is
instantiated, keeping the deprecated surface working while nudging callers
to the replacement.
"""

from __future__ import annotations

import functools
import warnings
from typing import Callable, Optional, TypeVar, Union

T = TypeVar("T", bound=Union[Callable, type])


def deprecated(reason: str, replacement: Optional[str] = None) -> Callable[[T], T]:
    """Mark a function or class as deprecated.

    Parameters
    ----------
    reason
        One-line explanation of why the object is deprecated.
    replacement
        Optional identifier of the recommended replacement, appended to the
        warning message as "Use <replacement> instead.".

    Returns
    -------
    Callable
        Decorator that wraps the target. For a function, the returned
        wrapper emits the warning on call. For a class, `__init__` is
        patched so the warning fires at construction (not at import),
        which keeps subclassing and isinstance checks clean.
    """

    def decorator(obj: T) -> T:
        msg = f"{obj.__module__}.{obj.__qualname__} is deprecated. {reason}"
        if replacement:
            msg += f" Use {replacement} instead."

        if isinstance(obj, type):
            orig_init = obj.__init__

            @functools.wraps(orig_init)
            def new_init(self, *args, **kwargs):
                warnings.warn(msg, DeprecationWarning, stacklevel=2)
                orig_init(self, *args, **kwargs)

            obj.__init__ = new_init  # type: ignore[assignment]
            return obj

        @functools.wraps(obj)  # type: ignore[arg-type]
        def wrapper(*args, **kwargs):
            warnings.warn(msg, DeprecationWarning, stacklevel=2)
            return obj(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator
