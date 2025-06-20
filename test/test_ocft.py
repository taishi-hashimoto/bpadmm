# %%
import numpy as np
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from os.path import join, dirname
from bpadmm import ocft
from antarrlib.decibel import dB

rng = np.random.default_rng(42)

N = 128
factor = 50
decimation = 1
K = factor * N // decimation

t = np.arange(N) / N
ot = np.arange(K) / K

f1 = 3.5
f2 = 5.5

ffttime = 0
ocftime = 0
pytime = 0

pfx = np.zeros(len(t))
pox = np.zeros(len(ot))

xx = np.array([
    np.exp(1j*2*np.pi*f1*t) + np.exp(1j*(2*np.pi*f2*t + np.pi/2)) +
    0.1 * (rng.normal(size=len(t)) + 1j*rng.normal(size=len(t)))
    for _ in range(50)])


now = time()
fx = np.fft.fft(xx, axis=-1)
ffttime += time() - now
pfx = np.sum(np.abs(np.fft.fftshift(fx, axes=-1))**2, axis=0)

now = time()
ox, f, info = ocft(
    xx, factor,
    threshold=(0.1, 0.1),
    maxiter=1000, stepiter=100, patience=10,
    info=True,
    axis=-1, decimation=decimation, dt=1/K,
    rng=rng)
ocftime += time() - now
pox += np.sum(np.abs(ox)**2, axis=0)

print(f"ffttime: {ffttime:.2g} s, ocftime: {ocftime:.2g} s")
print(info)

# %%

noi_pfx = np.nanmedian(pfx)
noi_pox = np.nanmedian(pox)
pox0 = np.where(pox < 1e-3, 0, pox)

fig, ax = plt.subplots(1, 1, figsize=(10, 3))
ax.grid()
ax.plot(
    np.fft.fftshift(np.fft.fftfreq(N, 1/N)),
    dB(pfx, "max"),
    ls="-", color="gray", label="FFT")
ax.set_ylim(ymin=dB(noi_pfx, np.max(pfx)) - 5)
# ax.set_autoscale_on(False)
ax.xaxis.set_major_formatter("{x:.0f} Hz")
ax.yaxis.set_major_formatter("{x:.0f} dB")
ii = np.argsort(f)
ax.plot(
    f[ii] / factor * decimation,
    dB(pox[ii], "max"),
    ls="-", color="red", label="OCFT")
ax.legend()
fig.tight_layout()

fig.savefig(join(dirname(__file__), "ocft.png"))

# %%
state = info.state
fig, axes = plt.subplots(2, 2, figsize=(10, 3))
ax = axes[0, 0]
ax.set_yscale("log")
ax.grid()
ax.plot(state.diff_x.T)
ax.set_title("Convergence of x")
ax = axes[0, 1]
ax.set_yscale("log")
ax.grid()
ax.plot(state.l1_norm.T)
ax.set_title("Convergence of l1")
ax = axes[1, 0]
ax.set_yscale("log")
ax.grid()
ax.plot(state.res_prim.T)
ax.set_title("Primal Residual")
ax = axes[1, 1]
ax.set_yscale("log")
ax.grid()
ax.plot(state.res_dual.T)
ax.set_title("Dual Residual")
fig.tight_layout()
# %%
