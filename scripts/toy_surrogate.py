"""A mock surrogate model, built with the three steps of the NR surrogates (Field+ 2014; Blackman+ 2015;
Varma+ 2019), on non-spinning TEOBResumS (2,2) waveforms that play the role of the NR simulations.

The steps (data, reduced basis, empirical interpolation, fits, test) are described where they are implemented.

References
  Field et al., Phys. Rev. X 4, 031006 (2014), https://arxiv.org/abs/1308.3565          (the method)
  Blackman et al., Phys. Rev. Lett. 115, 121102 (2015), https://arxiv.org/abs/1502.07758  (first NR surrogate)
  Varma et al., Phys. Rev. D 99, 064045 (2019), https://arxiv.org/abs/1812.07865        (NRHybSur3dq8)
  Varma et al., Phys. Rev. Research 1, 033015 (2019), https://arxiv.org/abs/1905.09300  (NRSur7dq4)

Usage
  python toy_surrogate.py         # build the model from the waveforms in data/; figures toy_sur_*
  python toy_surrogate.py extrap  # outside the training range; figure toy_sur_q20
  python toy_surrogate.py data    # regenerate missing waveforms (needs TEOBResumS: EOBRun_module)
"""

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from style import C, INK, MUTED, MARK, save, done, zoom_td  # noqa: E402
from wf_utils import eta_of, eob_22, mismatch_td  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
Q_TRAIN = [float(q) for q in np.round(np.linspace(1.0, 8.0, 20), 3)]
Q_TEST = [2.66, 4.87, 7.08]  # midpoints between training mass ratios: the hardest case
Q_EXTRA = [10.0, 12.0, 15.0, 20.0]  # outside the training range (extrapolation test)
F0 = 0.005  # initial GW frequency (M f): at least ~3000 M before the peak for q = 1
T0, T1, DT = -3000.0, 100.0, 1.0
TGRID = np.arange(T0, T1 + DT / 2, DT)
IPK, I100 = np.searchsorted(TGRID, 0.0), np.searchsorted(
    TGRID, -100.0
)  # the peak, 100 M before it
TOL_A = (
    1e-8  # greedy tolerances (worst relative squared L2 error over the training set)
)
TOL_PHI = 1e-10
DEG = 5  # degree of the fits in eta
N_TRAIN = [4, 6, 8, 10, 14, 20]  # sizes of the training set, for the last figure


# --------------------------------------------------------------------------------------------------
# Data
#
# EOB waveforms for 20 mass ratios in [1, 8] (training) and q = 2.66, 4.87, 7.08 (test, half-way between
# training points). Each is aligned so that the amplitude peaks at t = 0 and sampled on a common grid
# t in [-3000, 100] M (dt = 1 M). The (2,2) mode is decomposed into the slowly varying amplitude A(t) and
# phase phi(t), with phi(t = 0) = 0 (the NR surrogates work with the same kind of data, in the co-orbital frame).
# --------------------------------------------------------------------------------------------------
def cache(q):
    return DATA / f"toy_sur_q{q:g}.npz"


def generate(qs=None):
    DATA.mkdir(exist_ok=True)
    for q in qs or Q_TRAIN + Q_TEST:
        if not cache(q).exists():
            t, A, ph, n_dropped = eob_22(q, F0)
            np.savez_compressed(cache(q), t=t, A=A, phi=ph, q=q, n_dropped=n_dropped)
            print(f"q = {q:g}: {t[-1] - t[0]:.0f} M, dropped {n_dropped}")


@lru_cache(maxsize=None)
def load(q):
    """A(t), phi(t) on TGRID, peak at t = 0 (from a spline of the amplitude), phi(0) = 0."""
    d = np.load(cache(q))
    t, A, ph = d["t"], d["A"], d["phi"]
    sA = CubicSpline(t, A)
    i = np.argmax(A)
    tp = minimize_scalar(
        lambda x: -sA(x),
        bounds=(t[max(i - 3, 0)], t[min(i + 3, t.size - 1)]),
        method="bounded",
    ).x
    assert t[0] < tp + T0, f"q = {q}: waveform too short"
    tt = TGRID + tp
    Ag, pg = sA(tt), CubicSpline(t, ph)(tt)
    return Ag, pg - pg[IPK]


# --------------------------------------------------------------------------------------------------
# Step 1: reduced basis
#
# A greedy algorithm picks, one at a time, the training waveform worst represented by the current basis
# (L2 projection error over the time grid), until the error is below a tolerance. Separately for A and phi.
# --------------------------------------------------------------------------------------------------
class ReducedBasis:
    """Greedy reduced basis for the rows of X (Gram-Schmidt): orthonormal basis B, picked rows, errors."""

    def __init__(self, X, tol):
        norms = np.einsum("ij,ij->i", X, X)
        E = []
        picked = [int(np.argmax(norms))]
        B = [X[picked[0]] / np.sqrt(norms[picked[0]])]
        while True:
            Bm = np.array(B)
            R = X - (X @ Bm.T) @ Bm
            err = np.einsum("ij,ij->i", R, R) / norms
            E.append(err.max())
            if err.max() < tol or len(B) == X.shape[0]:
                break
            k = int(np.argmax(err))
            picked.append(k)
            v = R[k] - (R[k] @ Bm.T) @ Bm  # re-orthogonalise for stability
            B.append(v / np.linalg.norm(v))
        self.B, self.picked, self.err = np.array(B), picked, np.array(E)
        self.size = len(B)

    def project(self, x):
        """Projection of x(t) on the basis"""
        return (x @ self.B.T) @ self.B


# --------------------------------------------------------------------------------------------------
# Step 2: empirical interpolation (EIM)
#
# For each basis, the empirical time nodes T_i are chosen greedily so that the waveform is fixed by its
# values at the nodes: X(t) ~ sum_i B_i(t) X(T_i).
# --------------------------------------------------------------------------------------------------
class EmpiricalInterpolant:
    """Empirical interpolation nodes and interpolant matrix I for the orthonormal basis B (n x N)."""

    def __init__(self, B):
        nodes = [int(np.argmax(np.abs(B[0])))]
        for j in range(1, B.shape[0]):
            V = B[:j, nodes].T  # V[i, k] = B_k(T_i)
            c = np.linalg.solve(V, B[j, nodes])
            r = B[j] - c @ B[:j]
            nodes.append(int(np.argmax(np.abs(r))))
        self.nodes = np.array(nodes)
        self.I = np.linalg.solve(B[:, nodes], B)

    def __call__(self, values):
        """X(t) ~ sum_i I_i(t) X(T_i), from the values at the nodes"""
        return values @ self.I


# --------------------------------------------------------------------------------------------------
# Step 3: fits
#
# The values A(T_i; q) and phi(T_i; q) are fitted as functions of eta with a polynomial of degree DEG
# (least squares on the training points). Evaluation: the fits at the nodes, then the sum of step 2.
# --------------------------------------------------------------------------------------------------
class NodeFits:
    """Polynomial fits in x = eta of the values of X at the nodes."""

    def __init__(self, x, X, nodes, deg):
        self.coef = [np.polyfit(x, X[:, i], deg) for i in nodes]

    def __call__(self, x):
        """Values at the nodes, at any x"""
        return np.array([np.polyval(c, x) for c in self.coef])


class ComponentSurrogate:
    """Surrogate of one component of the waveform (A or phi): X[i] is the component for the i-th training
    waveform, x its eta."""

    def __init__(self, X, x, tol, deg):
        self.X = X
        self.basis = ReducedBasis(X, tol)  # step 1
        self.interp = EmpiricalInterpolant(self.basis.B)  # step 2
        self.fits = NodeFits(x, X, self.interp.nodes, deg)  # step 3

    def __call__(self, x):
        return self.interp(self.fits(x))


class Surrogate:
    def __init__(self, qs, tol_A=TOL_A, tol_phi=TOL_PHI, deg=DEG):
        self.qs = np.array(qs)
        x = eta_of(self.qs)
        deg = min(deg, len(qs) - 1)
        self.A = ComponentSurrogate(np.array([load(q)[0] for q in qs]), x, tol_A, deg)
        self.phi = ComponentSurrogate(
            np.array([load(q)[1] for q in qs]), x, tol_phi, deg
        )

    def __call__(self, q):
        """A(t), phi(t) on TGRID at mass ratio q"""
        return self.A(eta_of(q)), self.phi(eta_of(q))


# --------------------------------------------------------------------------------------------------
# Test
#
# The surrogate is evaluated at the test mass ratios and compared with EOB in the time domain.
# --------------------------------------------------------------------------------------------------
def errors(sur, q):
    A0, P0 = load(q)
    A1, P1 = sur(q)
    dP = P1 - P0
    w = TGRID < T0 + 1000  # align the phase over the first 1000 M, as for EOB/NR
    dP = dP - dP[w].mean()
    mm = mismatch_td(A0 * np.exp(-1j * P0), A1 * np.exp(-1j * P1))
    return dict(
        dphi_100=dP[I100],
        dphi_pk=dP[IPK],
        dphi_max=np.abs(dP[TGRID <= 0]).max(),
        dA_pk=A1[IPK] / A0[IPK] - 1,
        mm=mm,
    )


def report(sur, q, prefix=""):
    e = errors(sur, q)
    print(
        f"{prefix}q = {q:g}: dphi(-100M) = {e['dphi_100']:+.2e}, dphi(peak) = {e['dphi_pk']:+.2e}, "
        f"max|dphi| = {e['dphi_max']:.2e} rad, dA/A(peak) = {e['dA_pk']:+.1e}, mismatch = {e['mm']:.1e}"
    )


# --------------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------------
def fig_greedy(sur):
    """Greedy error against the size of the basis"""
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    for comp, mk, c, lab in (
        (sur.A, "o-", C[0], r"amplitude $A$"),
        (sur.phi, "s-", C[1], r"phase $\varphi$"),
    ):
        ax.semilogy(
            np.arange(1, comp.basis.err.size + 1),
            comp.basis.err,
            mk,
            color=c,
            ms=6,
            label=lab,
            **MARK,
        )
    ax.set(xlabel="number of basis elements", ylabel="greedy error (worst case)")
    ax.legend(loc="upper right")
    done(fig, "toy_sur_greedy")


def fig_eim(sur):
    """EIM nodes on a waveform"""
    A0, P0 = load(Q_TRAIN[0])
    fig, axes = plt.subplots(
        2, 1, figsize=(5.6, 4.4), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]}
    )
    axes[0].plot(TGRID, A0 * np.cos(P0), color=MUTED, lw=0.8)
    axes[0].plot(TGRID, A0, color=C[0], lw=1.6, label=r"$A(t)$")
    axes[0].plot(
        TGRID[sur.A.interp.nodes],
        A0[sur.A.interp.nodes],
        "o",
        color=C[0],
        ms=7,
        label="nodes for $A$",
        **MARK,
    )
    axes[0].set(ylabel=r"$\Re h_{22}$, $A$")
    axes[0].legend(loc="upper left", ncol=2)
    axes[1].plot(TGRID, P0, color=C[1], lw=1.6, label=r"$\varphi(t)$")
    axes[1].plot(
        TGRID[sur.phi.interp.nodes],
        P0[sur.phi.interp.nodes],
        "s",
        color=C[1],
        ms=7,
        label=r"nodes for $\varphi$",
        **MARK,
    )
    axes[1].set(
        xlabel=r"$(t-t_{\rm peak})/M$", ylabel=r"$\varphi$ [rad]", xlim=(T0, T1)
    )
    axes[1].legend(loc="upper left", ncol=2)
    done(fig, "toy_sur_eim")


def fig_fits(sur):
    """Fits of the node values across the parameter space"""
    x = eta_of(sur.qs)
    xx = np.linspace(x.min(), x.max(), 200)
    fig, axes = plt.subplots(2, 1, figsize=(4.8, 4.6), sharex=True)
    for ax, comp, lab, c in zip(axes, (sur.A, sur.phi), ("A", r"\varphi"), C):
        j = int(
            np.argmin(np.abs(TGRID[comp.interp.nodes] - (-50.0)))
        )  # a node close to merger
        node, coef = comp.interp.nodes[j], comp.fits.coef[j]
        ax.plot(x, comp.X[:, node], "o", color=c, ms=6, label="training data", **MARK)
        ax.plot(xx, np.polyval(coef, xx), color=INK, lw=1.4, label=f"degree-{DEG} fit")
        for q in Q_TEST:
            lab_q = "test" if q == Q_TEST[0] else None
            ax.plot(
                eta_of(q),
                np.polyval(coef, eta_of(q)),
                "D",
                color=C[2],
                ms=7,
                label=lab_q,
                **MARK,
            )
        ax.set_ylabel(rf"${lab}(T_i)$, $T_i={TGRID[node]:.0f}M$")
        ax.legend(loc="best")
    axes[-1].set_xlabel(r"$\eta$")
    done(fig, "toy_sur_fits")


def fig_td(sur):
    """Test waveforms in the time domain"""
    fig, axes = plt.subplots(len(Q_TEST), 1, figsize=(5.6, 5.4), sharex=True)
    for ax, q in zip(axes, Q_TEST):
        A0, P0 = load(q)
        A1, P1 = sur(q)
        ax.plot(TGRID, A0 * np.cos(P0), color=INK, lw=1.6, label="EOB")
        ax.plot(
            TGRID,
            A1 * np.cos(P1),
            color=C[1],
            lw=1.0,
            ls=(0, (3, 1.5)),
            label="mock surrogate",
        )
        ax.set_ylabel(r"$\Re h_{22}$")
        ax.text(
            0.02,
            0.88,
            rf"$q={q:g}$",
            transform=ax.transAxes,
            color=INK,
            fontsize=12,
            va="top",
        )
        ax.set_xlim(-600, T1)
    axes[-1].set_xlabel(r"$(t-t_{\rm peak})/M$")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.legend(h_, l_, loc="upper center", ncol=2, fontsize=10.5, frameon=False)
    save(fig, "toy_sur_td")
    plt.close(fig)


def fig_training():
    """Test error against the size of the training set (subsets of the training grid, same end points)"""
    out = []
    for n in N_TRAIN:
        idx = np.unique(np.round(np.linspace(0, len(Q_TRAIN) - 1, n)).astype(int))
        sur = Surrogate([Q_TRAIN[i] for i in idx], deg=min(DEG, n - 2))
        out.append(max(errors(sur, q)["mm"] for q in Q_TEST))
        print(f"training size {n}: worst test mismatch {out[-1]:.1e}")
    fig, ax = plt.subplots(figsize=(4.8, 3.4))
    ax.semilogy(N_TRAIN, out, "o-", color=C[0], ms=6, **MARK)
    ax.set(xlabel="number of training waveforms", ylabel="worst test mismatch")
    done(fig, "toy_sur_training")


def figures():
    sur = Surrogate(Q_TRAIN)
    nA, nP = sur.A.basis.size, sur.phi.basis.size
    print(
        f"training: {len(Q_TRAIN)} waveforms, q in [{Q_TRAIN[0]}, {Q_TRAIN[-1]}], {TGRID.size} time samples"
    )
    print(f"basis: {nA} for A, {nP} for phi; EIM nodes: {nA} + {nP}")
    print("greedy picks (q), phase:", [Q_TRAIN[k] for k in sur.phi.basis.picked])
    for q in Q_TEST:
        report(sur, q, "test ")
    fig_greedy(sur)
    fig_eim(sur)
    fig_fits(sur)
    fig_td(sur)
    fig_training()


def extrapolation():
    """Surrogate evaluated outside the training range 1 <= q <= 8. Figure: toy_sur_q20."""
    generate(Q_EXTRA)
    sur = Surrogate(Q_TRAIN)
    for q in Q_EXTRA:
        report(sur, q)
    (A0, P0), (A1, P1) = load(20.0), sur(20.0)
    h0 = A0 * np.cos(P0)
    fig, (a1, a2) = zoom_td(
        TGRID,
        h0,
        A1 * np.cos(P1),
        "EOB, $q=20$",
        "mock surrogate (extrapolated)",
        (5.4, 3.9),
    )
    lim = 1.25 * np.abs(h0[(TGRID > -1200) & (TGRID < 60)]).max()
    a1.set_ylim(-lim, lim)
    done(fig, "toy_sur_q20")


if __name__ == "__main__":
    if sys.argv[1:] == ["data"]:
        generate()
    elif sys.argv[1:] == ["extrap"]:
        extrapolation()
    else:
        figures()
