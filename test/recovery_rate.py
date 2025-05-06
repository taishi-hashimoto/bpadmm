# %% Something like Donoho-Tanner phase transition plot of the basis pursuit problem
from os.path import join, dirname
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from bpadmm import basis_pursuit_admm

nseeds = 10  # Number of random seeds.
p = 100  # Number of parameters of X.

results = []
with tqdm(total=nseeds * p) as pbar:
    for seed in range(nseeds):
        rng = np.random.default_rng(seed)
        for n in range(1, p+1):  # Number of observations.
            A = rng.normal(size=(n, p))  # Observation matrix.
            x = []
            y = []
            for s in range(1, p+1):  # Number of non-zero elements.
                x0 = np.hstack((rng.normal(size=s), np.zeros(shape=p - s)))
                x0 = rng.permutation(x0)
                y.append(A @ x0)
                x.append(x0)
            x = np.c_[x]
            y = np.c_[y]
            x1, info = basis_pursuit_admm(
                A, y, threshold=np.linalg.norm(A, np.inf) * 1e-4,
                info=True)
            mse = np.sum((x1.real - x)**2, axis=-1)
            for i, s in enumerate(range(1, p+1)):
                results.append({
                    "seed": seed, "n": n, "s": s,
                    "niters": info["i"][i],
                    "mse_true": mse[i],
                    "mse_calc": info["mse"][i][info["i"][i]],
                })
            pbar.update(1)
            pbar.set_description(f"seed={seed}, n={n}, y.shape={y.shape}")

df = pd.DataFrame(results)
# %%
df_ave = df.groupby(["n", "s"]).mean()
df_ave.sort_index(inplace=True)
# Recovery rate
recovery_rate = df.groupby(["n", "s"]).mse_true.aggregate(lambda x: (x < 1e-6).mean()).values.reshape(n, s) * 100
# %% Theoretical curve from below notebook:
# https://statweb.rutgers.edu/PCB71/donoho_tanner_phase_transition.html
from scipy.stats import norm

f = norm.pdf
Phi = norm.cdf

F =     lambda t: 2*( (1+t**2)*norm.cdf(-t) - t*norm.pdf(t) )
delta = lambda t: 4*norm.pdf(t)/(2*( t*(1-2*norm.cdf(-t)) + 2*norm.pdf(t) ))
s_max = lambda t: (delta(t) - F(t))/( (1+t**2) - F(t))

m = plt.imshow(recovery_rate, extent=(0, p, 0, p), origin="lower")
g = np.linspace(0.02, 10, 1000)
plt.plot(s_max(g)*p, delta(g)*p, "r-", lw=2, zorder=10)
plt.colorbar(m, format="%.0f%%", label="Recovery rate")
plt.xlabel("No. non-zero elements")
plt.ylabel("No. observations")
plt.title(f"No. tries = {nseeds}")
plt.tight_layout()
plt.savefig(join(dirname(__file__), "recovery_rate.png"))

# %%
niters = np.reshape(df_ave["niters"].astype(int), [n, s])
m = plt.imshow(niters, extent=(0, p, 0, p), origin="lower")
plt.colorbar(m, label="No. Iterations")
plt.xlabel("No. non-zero elements")
plt.ylabel("No. observations")
plt.tight_layout()
# %%
mse_true = np.reshape(df_ave["mse_true"].astype(float), [n, s])
m = plt.imshow(mse_true, extent=(0, p, 0, p), origin="lower", vmax=1e-6)
plt.colorbar(m, label="MSE (True)")
plt.xlabel("No. non-zero elements")
plt.ylabel("No. observations")
plt.tight_layout()

# %%
mse_calc = np.reshape(df_ave["mse_calc"].astype(float), [n, s])
m = plt.imshow(mse_calc, extent=(0, p, 0, p), origin="lower", vmax=1e-10)
plt.colorbar(m, label="MSE (Calc)")
plt.xlabel("No. non-zero elements")
plt.ylabel("No. observations")
plt.tight_layout()

# %%
