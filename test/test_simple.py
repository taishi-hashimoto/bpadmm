# %% Simple example.
# http://www-adsys.sys.i.kyoto-u.ac.jp/mohzeki/Presentation/lecturenote20150902.pdf
 
import numpy as np
import matplotlib.pyplot as plt
from bpadmm import basis_pursuit_admm
from bpadmm.reference import basis_pursuit_admm as bpadmm_python
from os.path import join, dirname


N = 100
M = 50
K = 10
A = np.random.normal(size=(M, N))
x0 = np.vstack((np.random.normal(size=(K, 1)), np.zeros(shape=(N-K, 1))))
x0 = np.random.permutation(x0)

y = A @ x0
# %%

threshold = np.linalg.norm(A) * 0.001

x, info = basis_pursuit_admm(A+0j, y+0j, threshold=threshold)
# NOTE: xtol is set to very small value because basis_pursuit_admm doesn't
#       check it.
trace = bpadmm_python(A+0j, y+0j, threshold=threshold, trace=True, xtol=1e-16)

fig, ax = plt.subplots()
ax.stem(x0.real, linefmt="k-", markerfmt="ko")
# ax.plot(x0, c="k")
ax.stem(x.real, linefmt="none", markerfmt="rx")

fig.tight_layout()
fig.savefig(join(dirname(__file__), "simple.png"))

# %%

x_nonzero = np.count_nonzero(np.abs(trace["x"]) > 1e-3, axis=0)
z_nonzero = np.count_nonzero(np.abs(trace["z"]) > 1e-3, axis=0)
u_nonzero = np.count_nonzero(np.abs(trace["u"]) > 1e-3, axis=0)

x_l1 = np.sum(np.abs(trace["x"]), axis=0)
z_l1 = np.sum(np.abs(trace["z"]), axis=0)
u_l1 = np.sum(np.abs(trace["u"]), axis=0)

x_norm = np.linalg.norm(trace["x"], axis=0)
z_norm = np.linalg.norm(trace["z"], axis=0)
u_norm = np.linalg.norm(trace["u"], axis=0)

fig, axes = plt.subplots(3, 3, figsize=(10, 6), sharex=True)
axes[0, 0].plot(x_nonzero)
axes[0, 1].plot(z_nonzero)
axes[0, 2].plot(u_nonzero)
axes[0, 0].set_title(r"$\mathbf{x}$")
axes[0, 1].set_title(r"$\mathbf{z}$")
axes[0, 2].set_title(r"$\mathbf{u}$")
axes[1, 0].plot(x_l1)
axes[1, 1].plot(z_l1)
axes[1, 2].plot(u_l1)
axes[2, 0].plot(x_norm)
axes[2, 1].plot(z_norm)
axes[2, 2].plot(u_norm)
axes[0, 0].set_ylabel(r"$|| \cdot ||_0$")
axes[1, 0].set_ylabel(r"$|| \cdot ||_1$")
axes[2, 0].set_ylabel(r"$|| \cdot ||_2^2$")
# axes[1, 0].set_yscale("log")
# axes[2, 0].set_yscale("log")
fig.tight_layout()
fig.savefig(join(dirname(__file__), "reference_convergence.png"))

# %%
