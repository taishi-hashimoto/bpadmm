import matplotlib.pyplot as plt
from matplotlib.colorbar import Colorbar
import numpy as np
from os.path import join, dirname
from bpadmm._impl import _soft_threshold


def test_soft_threshold_real():
    x = np.linspace(-9, 9, 200)
    y = _soft_threshold(x + 0j, 4).real  # workaround for real numbers...

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(x, y, "r-", label=r"$S_{\lambda = 4}(x)$")
    ax.axline((0, 0), slope=1, color="k", ls=":", zorder=-1)
    # ax.axvline(4, color="k", ls=":", zorder=-1)
    # ax.axvline(-4, color="k", ls=":", zorder=-1)
    ax.set_aspect("equal", "box")
    ax.set_xlim(min(x), max(x))
    ax.legend()
    ax.grid()
    fig.tight_layout()
    fig.savefig(join(dirname(__file__), "soft_threshold_real.png"))


def test_soft_threshold_complex():
    x = np.linspace(-10, 10)[:, None] + 1j * np.linspace(-10, 10)[None, :]
    cmap = plt.get_cmap("seismic", lut=11)
    values = _soft_threshold(x, 4)

    fig, ((ax_re, ax_im), (cax_re, cax_im)) = plt.subplots(
        2, 2,
        gridspec_kw=dict(height_ratios=(20, 1)),
        subplot_kw=dict(sharex="row", sharey="row"), figsize=(8, 5))
    m_re = ax_re.pcolormesh(x.real, x.imag, values.real, cmap=cmap)
    ax_re.set_aspect("equal", "box")
    ax_re.set_title(r"$\mathrm{Re}\ [ S_{\lambda = 4}(\mathbf{x})]$")
    Colorbar(ax=cax_re, mappable=m_re, orientation="horizontal")
    m_im = ax_im.pcolormesh(x.real, x.imag, values.imag, cmap=cmap)
    Colorbar(ax=cax_im, mappable=m_im, orientation="horizontal")
    ax_im.set_aspect("equal", "box")
    ax_im.set_title(r"$\mathrm{Im}\ [ S_{\lambda = 4}(\mathbf{x})]$")
    fig.tight_layout()
    fig.savefig(join(dirname(__file__), "soft_threshold_complex.png"))
