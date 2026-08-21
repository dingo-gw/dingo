from .random_fd_modes import random_inspiral_FD_modes

# LAL-based mode functions (require lalsimulation at import time)
try:
    from .lalsimulation_simInspiralChooseFDModes import lalsim_inspiral_choose_FD_modes
    from .lalsimulation_simInspiralChooseTDModes import lalsim_inspiral_choose_TD_modes
except ImportError:
    pass

# GWSignal-based mode functions (require gwsignal + pyseobnr)
try:
    from .gwsignal_generateFDModes import gwsignal_generate_FD_modes
    from .gwsignal_generateTDModes import gwsignal_generate_TD_modes
    from .gwsignal_generateTDModes_SEOBNRv5 import gwsignal_generate_TD_modes_SEOBNRv5
except ImportError:
    pass
