import numpy as np
import time
from bpadmm._impl import _pinv


def test_pinv():
    a = np.random.normal(size=(25, 30000))
    t0 = time.time()
    a1 = np.linalg.pinv(a)
    print("np.linalg.pinv", time.time() - t0)
    t0 = time.time()
    a2 = _pinv(a + 0j)
    print("bpadmm._impl._pinv", time.time() - t0)
    assert np.allclose(a1, a2)
