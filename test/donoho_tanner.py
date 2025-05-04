# %% Donoho-Tanner phase transition plot
from os.path import join, dirname
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from bpadmm import basis_pursuit_admm

nseeds = 10

p = 100  # Number of parameters of X.
results = []
with tqdm(total=nseeds * p) as pbar:
    for seed in range(nseeds):
        rng = np.random.default_rng(seed)
        for n in range(1, p+1):
            A = rng.normal(size=(n, p))  # Observation matrix.
            x = []
            y = []
            for s in range(1, p+1):
                x0 = np.hstack((rng.normal(size=s), np.zeros(shape=p - s)))
                x0 = rng.permutation(x0)
                y.append(A @ x0)
                x.append(x0)
            x = np.c_[x]
            y = np.c_[y]
            x1, info = basis_pursuit_admm(A, y, threshold=np.linalg.norm(A) * 1e-4)
            result = [np.allclose(a, b, atol=1e-6, rtol=1e-4) for a, b in zip(x1.real, x)]
            results.append((seed, n, result))
            pbar.update(1)
            pbar.set_description(f"seed={seed}, n={n}, y.shape={y.shape}")

# %%
from collections import defaultdict

# Collect results for each n.
collected_by_n = defaultdict(list)
for seed, n, result in results:
    collected_by_n[n].append(result)

recovery_rate = np.sum(list(collected_by_n.values()), axis=1) / nseeds * 100

m = plt.imshow(recovery_rate, extent=(0, p, 0, p), origin="lower")
plt.colorbar(m, format="%.0f%%", label="Recovery rate")
plt.xlabel("Sparsity (No. non-zero elements)")
plt.ylabel("No. observations")
plt.tight_layout()
plt.savefig(join(dirname(__file__), "donoho_tanner.png"))
# %%
