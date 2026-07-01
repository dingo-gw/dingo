import bilby.core.utils.random
import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _seed_rngs():
    """Seed numpy and bilby RNGs before every test.

    Makes the full suite deterministic: each test starts from the same known
    state, independent of test ordering. Avoids stochastic failures like the
    ones tracked in issue #377.
    """
    np.random.seed(0)
    bilby.core.utils.random.seed(0)
