import numpy as np
import pytest
# from dingo.gw.prior import Uniform, Sine, Cosine


# @pytest.mark.parametrize(
#     "X",
#     [
#         pytest.param(
#             Uniform(minimum=np.random.randint(5, 50), maximum=np.random.randint(51, 200)), id="uniform"
#         ),
#         pytest.param(
#             Sine(), id="sine"
#         ),
#         pytest.param(
#             Cosine(), id="cosine"
#         ),
#     ],
# )

@pytest.mark.obsolete
def test_analytical_mean_std(X):
    N = 100000
    s = X.sample(size=N)
    assert np.abs(X.mean() - np.mean(s)) < 10*np.std(s)/np.sqrt(N)
    assert np.abs(X.std() - np.std(s)) < 5*np.std(s)/np.sqrt(N)


