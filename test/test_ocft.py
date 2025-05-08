# %%
import numpy as np
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
from os.path import join, dirname
from bpadmm import ocft
from antarrlib.decibel import dB


N = 128
factor = 10
decimation = 2
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
    0.1 * (np.random.normal(size=len(t)) + 1j*np.random.normal(size=len(t)))
    for _ in range(50)])


now = time()
fx = np.fft.fft(xx, axis=-1)
ffttime += time() - now
pfx = np.sum(np.abs(np.fft.fftshift(fx, axes=-1))**2, axis=0)

now = time()
ox, f, info = ocft(xx, factor, maxiter=10000, stepiter=1, patience=10, info=True, axis=-1, decimation=decimation, dt=1/K)
ocftime += time() - now
pox += np.sum(np.abs(ox)**2, axis=0)

print(ffttime, ocftime)
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
    ls="-", color="gray")
ax.set_ylim(ymin=dB(noi_pfx, np.max(pfx)) - 5)
# ax.set_autoscale_on(False)
ax.set_ylabel("Spectral Density [dB]")
ax.set_xlabel("Freqeuncy [Hz]")
fig.tight_layout()
ii = np.argsort(f)
ax.plot(
    f[ii] / factor * decimation,
    dB(pox[ii], "max"),
    ls="-", color="red")
# ax.set_xlim(xmin=-25, xmax=25)

fig.savefig(join(dirname(__file__), "bpadmm_dft.png"))

# %%
