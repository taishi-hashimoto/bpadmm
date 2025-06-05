# %% Simple example.
# http://www-adsys.sys.i.kyoto-u.ac.jp/mohzeki/Presentation/lecturenote20150902.pdf
 
import numpy as np
import matplotlib.pyplot as plt
from bpadmm import basis_pursuit_admm
from os.path import join, dirname

N = 100  # The number of parameters.
M = 50  # The number of observations.
K = 10  # The number of non-zero elements.

A = np.random.normal(size=(M, N))  # Observation matrix.

# Original signal.
x0 = np.r_[np.random.normal(size=K), np.zeros(shape=N - K)]
x0 = np.random.permutation(x0)

# Measurement.
y = A @ x0
# %%

threshold = 0.001  # Soft threashold.

result = basis_pursuit_admm(A+0j, y+0j, threshold=threshold)

x = result.x

fig, ax = plt.subplots()
ax.stem(x0.real, linefmt="k-", markerfmt="ko")
# ax.plot(x0, c="k")
ax.stem(x.real, linefmt="none", markerfmt="rx")

fig.tight_layout()
fig.savefig(join(dirname(__file__), "simple.png"))

# %% Check plot how computation went.
# %%
state = result.state
fig, axes = plt.subplots(2, 2)
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
fig.savefig(join(dirname(__file__), "convergence.png"))
# %%
