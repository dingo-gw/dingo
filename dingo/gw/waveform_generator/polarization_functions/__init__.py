from .random_fd import random_inspiral_FD

# LAL-based polarization functions (require lalsimulation at import time)
try:
    from .lalsimulation_simInspiralFD import lalsim_inspiral_FD
    from .lalsimulation_simInspiralTD import lalsim_inspiral_TD
except ImportError:
    pass

# GWSignal-based polarization functions (require gwsignal + pyseobnr)
try:
    from .gwsignal_generateFDWaveform import gwsignal_generate_FD_modes
    from .gwsignal_generateTDWaveform import gwsignal_generate_TD_modes
except ImportError:
    pass
