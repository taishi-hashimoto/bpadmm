# %%
import numpy as np
import matplotlib.pyplot as plt
from os.path import join, dirname
from antarrlib import freq2wlen, wlen2wnum, steering_vector, radial, dB
from antarrlib.noise import noise
from bpadmm import basis_pursuit_admm
from bpadmm.reference import basis_pursuit_admm as basis_pursuit_admm_python
import time


frequency = 47e6
wavelength = freq2wlen(frequency)
wavenumber = wlen2wnum(wavelength)


m = 5
offset_1d = np.arange(m) * wavelength/2
x_g, y_g = np.meshgrid(offset_1d, offset_1d)
M = m*m

r = np.c_[x_g.ravel(), y_g.ravel(), np.zeros_like(x_g.ravel())]

plt.plot(x_g.ravel(), y_g.ravel(), "ko")

# %%

ZE = np.linspace(0, 90, 91)
AZ = np.linspace(-180, 180, 361)

TARGETS = [
    (1, (15, 40)),
    (1, (10, -120)),
    (1, (20, 110)),
]

ze_g, az_g = np.meshgrid(np.deg2rad(ZE), np.deg2rad(AZ), indexing="ij")
v = radial(ze_g, az_g)

N = 5

x = noise((N, M), power=0.01)
for power, (zenith, azimuth) in TARGETS:
    # sig_1 = np.random.normal(size=N)
    sig_1 = power * np.ones(shape=N)
    x += sig_1[:, None] * steering_vector(
        wavenumber, r, radial(np.deg2rad(zenith), np.deg2rad(azimuth)), -1)

w = steering_vector(wavenumber, r, v, -1)

y = w.conjugate().dot(x.transpose())

fig, ax = plt.subplots(subplot_kw=dict(projection="polar"), figsize=(12, 10))

for power, (zenith, azimuth) in TARGETS:
    ax.plot(np.deg2rad(azimuth), zenith, "rx", mfc="none", ms=5)
ax.set_rlim(0, 50)
ax.pcolormesh(
    np.deg2rad(AZ), ZE,
    dB(np.mean(np.abs(y)**2, axis=-1), "max").reshape(ze_g.shape),
    vmin=-20, vmax=0)
# ax.grid()
fig.tight_layout()
fig.savefig(join(dirname(__file__), "bpadmm_fourier_imaging.png"))


# %%

A = w.T
y = x.T
tol = 1e-7

max_sigma = np.linalg.norm(A)
lambda_ = 1e-4 * max_sigma

NITER = 10000

t0 = time.time()
result, info = basis_pursuit_admm(
    A, x, threshold=lambda_,
    maxiter=NITER, stepiter=100)
result = np.atleast_2d(result).T
rstime = time.time() - t0

t0 = time.time()
result1 = []
for ii in range(N):
    y_ = x[ii, :, None]
    x_ = basis_pursuit_admm_python(
        A, y_, threshold=lambda_,
        maxiter=NITER, miniter=1000,
        xtol=tol, ftol=tol)
    result1.append(x_.ravel())
result1 = np.atleast_2d(result1).T
pytime = time.time() - t0

print(pytime, rstime)

# %%
yy = np.sum(np.abs(np.c_[result])**2, axis=-1)
yy[yy < 1e-4] = np.nan
fig, ax = plt.subplots(subplot_kw=dict(projection="polar"), figsize=(12, 10))
ax.set_rlim(0, 90)
ax.pcolormesh(
    np.deg2rad(AZ), ZE,
    10*np.log10(yy.reshape(ze_g.shape)/np.nanmax(yy)), vmin=-20)
for power, (zenith, azimuth) in TARGETS:
    ax.plot(np.deg2rad(azimuth), zenith, "ro", mfc="none", ms=20)
ax.set_facecolor("k")
ax.set_rlim(0, 50)
# ax.grid()
fig.tight_layout()
fig.savefig(join(dirname(__file__), "bpadmm_sparse_imaging.png"))

# %%
