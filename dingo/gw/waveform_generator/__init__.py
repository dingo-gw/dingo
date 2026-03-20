from .waveform_generator import (
    WaveformGenerator,
    NewInterfaceWaveformGenerator,
    generate_waveforms_parallel,
    sum_contributions_m,
)

# New-style API (ported from dingo-waveform)
from .new_api import (
    NewWaveformGenerator,
    RandomWaveformGenerator,
    LALSimWaveformGenerator,
    build_waveform_generator,
)
from .polarizations import (
    Polarization,
    BatchPolarizations,
    PolarizationProtocol,
    get_polarizations_from_fd_modes_m as new_get_polarizations_from_fd_modes_m,
    sum_contributions_m as new_sum_contributions_m,
)
from .waveform_parameters import (
    WaveformParameters,
    BBHWaveformParameters,
    RandomWaveformParameters,
    build_waveform_parameters,
)
from .waveform_generator_parameters import WaveformGeneratorParameters
