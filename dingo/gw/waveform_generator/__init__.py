from .api import (
    WaveformGenerator,
    RandomWaveformGenerator,
    LALSimWaveformGenerator,
    build_waveform_generator,
)
from .polarizations import (
    Polarization,
    BatchPolarizations,
    PolarizationProtocol,
    get_polarizations_from_fd_modes_m,
    sum_contributions_m,
)
from .waveform_parameters import (
    WaveformParameters,
    BBHWaveformParameters,
    RandomWaveformParameters,
    build_waveform_parameters,
)
from .waveform_generator_parameters import WaveformGeneratorParameters

# Deprecated legacy names re-exported at the historic top-level so old scripts
# still resolve — importing / calling them emits a DeprecationWarning.
# The legacy dict-based `WaveformGenerator` lives under
# `dingo.gw.waveform_generator.legacy` because it would otherwise shadow the
# natural class above.
from .legacy import (
    NewInterfaceWaveformGenerator,
    generate_waveforms_parallel,
)
