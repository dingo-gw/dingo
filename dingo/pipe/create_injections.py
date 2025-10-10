import functools

import numpy as np

import bilby
import bilby_pipe.input as input_mod
import bilby_pipe.create_injections as ci

_bilby_pipe_create_injection_file = ci.create_injection_file


@functools.wraps(_bilby_pipe_create_injection_file)
def create_injection_file(*args, **kwargs):
    # Monkey-patch create_injection_file() to allow for an offset in the time. This is
    # needed if the DINGO network time prior is not centered at 0.0, e.g., if we have a
    # Dirac delta prior.

    Toffset = kwargs.pop("Toffset", 0.0)

    if kwargs.get("trigger_time") is not None:
        kwargs["trigger_time"] += Toffset
    if kwargs.get("gpstimes") is not None:
        gpstimes = kwargs["gpstimes"]
        kwargs["gpstimes"] = (
            (np.asarray(gpstimes) + Toffset).tolist()
            if isinstance(gpstimes, list)
            else gpstimes + Toffset
        )

    return _bilby_pipe_create_injection_file(*args, **kwargs)


create_injection_file.__doc__ = (
    "Add a `Toffset` keyword to offset the trigger and GPS times before creating injections.\n\n"
    + _bilby_pipe_create_injection_file.__doc__
)

ci.create_injection_file = create_injection_file

_bilby_pipe_get_time_prior = input_mod.get_time_prior


@functools.wraps(_bilby_pipe_get_time_prior)
def get_time_prior(time, uncertainty, name="geocent_time", latex_label="$t_c$"):
    # Monkey-patch to allow for DeltaFunction priors.
    if uncertainty == 0.0:
        return bilby.core.prior.DeltaFunction(
            peak=time,
            name=name,
            latex_label=latex_label,
            unit="$s$",
        )
    elif uncertainty > 0.0:
        return _bilby_pipe_get_time_prior(
            time, uncertainty, name=name, latex_label=latex_label
        )
    else:
        raise ValueError(f"Time uncertainty {uncertainty} < 0.0.")


get_time_prior.__doc__ = (
    "A bilby.core.prior.DeltaFunction for the time parameter, if uncertainty == 0.0.\n\n"
    + _bilby_pipe_get_time_prior.__doc__
)


input_mod.get_time_prior = get_time_prior
