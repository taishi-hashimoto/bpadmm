# bpadmm

This is a Python library that implements a solver for the Compressed Sensing on underdetermined linear systems.
The algorithm is based on the Basis Pursuit (BP) with Alternating Direction Method of Multipliers (ADMM).

## Methodologies

Basis pursuit problem is defined as following sparse modeling:

```
minimize |x|_1
    x
subject to y = Ax
```

ADMM solves this problem by the following iterations:

```Python
A1 = A^T(A A^T)^(-1)  # pseudo inverse
x = z - u + A1 @ (y - A @ (z - u))
z = soft_threshold(x + u, threshold)
u += x - z
```

Methodologies are briefly explained in [doc/intro_cs.ipynb](doc/intro_cs.ipynb).

## Installation 

### Dependencies

- [jax](https://docs.jax.dev/en/latest/index.html)
- [numpy](https://pypi.org/project/numpy/)
- [tqdm](https://pypi.org/project/tqdm/)

Tests also need [matplotlib](https://pypi.org/project/matplotlib/) and [antarrlib](https://github.com/taishi-hashimoto/python-antarrlib).

### Installation

Use pip:

```
pip install .
```

## Usage

Import `bpadmm` Python package.

```Python

from bpadmm import basis_pursuit_admm

# n: The number of measurements in a single snapshot.
# p: The number of parameters in A.
# k: The number of snapshots.
# A: 2-d matrix with the size (n, p).
# y: 2-d matrix with the size (n, k).
# x: 2-d matrix with the size (n, k).

x, info = basis_pursuit_admm(
    A,  # Observation matrix.
    y,  # Observation (measurement).
    threshold,  # Soft threshold.
    # ... and some optional arguments ...
    info=True,
)

```

## Examples

You can find some more examples under [test](./test) directory.

#### Sparse modeling example in radar imaging

- [test/test_radar_imaging.py](test/test_radar_imaging.py)  

|Nonadaptive beamforming                       |Sparse modeling                         |
|----------------------------------------------|----------------------------------------|
|![nonadaptive](doc/bpadmm_fourier_imaging.png)|![bpadmm](doc/bpadmm_sparse_imaging.png)|


