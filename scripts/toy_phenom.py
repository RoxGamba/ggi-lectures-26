"""A mock non-spinning phenomenological model based on IMRPhenomXAS (Pratten+ 2020).

TEOBResumS (2,2) waveforms play the role of the EOB-NR hybrids. Units G = c = M = 1, frequencies are GW
frequencies Mf. The steps (data, fit of one waveform, fits across parameter space, comparison) are described
where they are implemented.

References
  IMRPhenomXAS: Pratten et al., Phys. Rev. D 102, 064001 (2020), https://arxiv.org/abs/2001.11412
  IMRPhenomD:   Husa et al., Phys. Rev. D 93, 044006 (2016), https://arxiv.org/abs/1508.07250
                Khan et al., Phys. Rev. D 93, 044007 (2016), https://arxiv.org/abs/1508.07253
  TaylorF2:     Buonanno et al., Phys. Rev. D 80, 084043 (2009), https://arxiv.org/abs/0907.0700

Usage
  python toy_phenom.py         # build the model from the waveforms in data/; figures toy_phenom_regions,
                               # toy_phenom_coeffs, toy_phenom_fd, toy_phenom_td
  python toy_phenom.py extrap  # inside and outside the training range; figures toy_phenom_extrap, toy_phenom_q20
  python toy_phenom.py data    # regenerate the waveforms (needs TEOBResumS: EOBRun_module)
"""

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import CubicSpline
from scipy.optimize import curve_fit

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from style import C, INK, MUTED, MARK, done, zoom_td  # noqa: E402
from wf_utils import eta_of, eob_22, match, align  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
Q_TRAIN = [1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0]
Q_TEST = 5.0
Q_EXTRA = [
    1.25,
    2.5,
    3.5,
    5.0,
    7.0,
    10.0,
    12.0,
    15.0,
    20.0,
]  # inside and outside the training range
F0, DT = 0.002, 0.5  # EOB initial GW frequency; sampling step of the cached waveforms
FMIN, FMAX = 0.004, 0.2  # band that is modelled
DEG = 2  # polynomial degree in eta for the parameter-space fits
NSHIFT = (
    1 << 20
)  # size of the FFT over the time shift in the match: resolution 1 / (NSHIFT df)
DASH = dict(
    color=C[1], lw=1.6, ls=(0, (4, 2))
)  # the model, in the frequency-domain figures


def cache(q):
    return DATA / f"toy_phenom_q{q:g}.npz"


# --------------------------------------------------------------------------------------------------
# Step 0: data
#
# h22(t) is Fourier transformed (peak at t = 0, Planck taper at the start). We use
#     A(f) = |h~(f)|,   phi'(f) = d arg h~ / df = -2 pi t(f),
# with h~ the transform of conj(h22) (positive frequencies) and phi' computed exactly as Im(h~' h~*)/|h~|^2,
# h~' being the transform of -2 pi i t h(t): no finite differences.
# f_RD and f_damp are measured from the ringdown of each waveform (fit of a damped sinusoid), and then fitted
# in eta (step 2). This is a shortcut: XAS takes them from the quasi-normal mode of the remnant, i.e. NR fits
# of the final mass and spin (Jimenez-Forteza et al. 2017, https://arxiv.org/abs/1611.00332) and rational
# fits of the Kerr quasi-normal frequencies as functions of the final spin (data of Berti, Cardoso & Starinets
# 2009, https://arxiv.org/abs/0905.2975; simple fits in Berti, Cardoso & Will 2006,
# https://arxiv.org/abs/gr-qc/0512160, used in toy_kerr/hamiltonian.py).
# --------------------------------------------------------------------------------------------------
class Waveform:
    """
    A TEOBResumS (2,2) waveform from the cache: h22(t) with the peak at t = 0, its ringdown frequencies
    f_RD, f_damp and its Fourier transform f, H(f), phi'(f).
    """

    def __init__(self, q):
        self.q, self.eta = q, eta_of(q)
        d = np.load(cache(q))
        t, A, ph = d["t"], d["A"], d["phi"]
        self.t = t - t[np.argmax(A)]
        self.h = A * np.exp(-1j * ph)  # h22 = A e^{-i phi}
        self.dt = self.t[1] - self.t[0]
        self.fRD, self.fdamp = self.ringdown()
        self.f, self.H, self.dphi = self.to_fd()

    @staticmethod
    def generate(qs=None):
        """Compute the waveforms with TEOBResumS and cache them."""
        DATA.mkdir(exist_ok=True)
        for q in qs or Q_TRAIN + [Q_TEST]:
            t, A, ph, n_dropped = eob_22(q, F0)
            tu = np.arange(t[0], t[-1], DT)  # uniform grid, for the FFT
            A, ph = CubicSpline(t, A)(tu), CubicSpline(t, ph)(tu)
            np.savez_compressed(cache(q), t=tu, A=A, phi=ph, q=q, n_dropped=n_dropped)
            print(f"q = {q:g}: {tu[-1] - tu[0]:.0f} M, {tu.size} samples")

    def ringdown(self, t1=10.0, t2=50.0):
        """f_RD, f_damp from a straight-line fit of log|h| and arg h after the peak."""
        t, h = self.t, self.h
        k = (t > t1) & (t < t2)
        om = -np.polyfit(t[k], np.unwrap(np.angle(h[k])), 1)[0]
        gam = -np.polyfit(t[k], np.log(np.abs(h[k])), 1)[0]
        return om / (2 * np.pi), gam / (2 * np.pi)

    def to_fd(self, ntaper_M=1500.0):
        """Fourier transform of conj(h22): f, h~(f), phi'(f) on the FFT frequencies around the band."""
        t, dt = self.t, self.dt
        g = np.conj(self.h) * planck(t.size, int(ntaper_M / dt))
        n = 1 << int(np.ceil(np.log2(t.size)) + 1)
        f = np.fft.fftfreq(n, dt)
        k = (f > 0.5 * FMIN) & (f < 1.2 * FMAX)
        f = f[k]
        ph0 = np.exp(
            -2j * np.pi * f * t[0]
        )  # the FFT assumes the series starts at t = 0
        H = np.fft.fft(g, n)[k] * dt * ph0
        Hp = np.fft.fft(-2j * np.pi * t * g, n)[k] * dt * ph0  # d H / d f
        return f, H, np.imag(Hp * np.conj(H)) / np.abs(H) ** 2


def planck(n, m):
    """Planck taper rising over the first m of n samples."""
    w = np.ones(n)
    x = np.arange(1, m)
    w[0] = 0.0
    with np.errstate(over="ignore"):
        w[1:m] = 1 / (1 + np.exp(m / x - m / (m - x)))
    return w


# --------------------------------------------------------------------------------------------------
# Post-Newtonian input: the TaylorF2 phase at 3.5PN, Eq. (3.18) of Buonanno, Iyer, Ochsner, Pan &
# Sathyaprakash (2009), https://arxiv.org/abs/0907.0700 (non-spinning; the constant v_lso in the 2.5PN
# logarithm is dropped: it does not contribute to phi')
# --------------------------------------------------------------------------------------------------
@lru_cache(maxsize=None)
def _tf2():
    """-d/df of the TaylorF2 3.5PN phase (non-spinning), without the linear (t_c) term: from sympy, once."""
    import sympy as sp

    R = sp.Rational
    F, n = sp.symbols("f eta", positive=True)
    v = (sp.pi * F) ** R(1, 3)
    p = [
        1,
        0,
        R(3715, 756) + R(55, 9) * n,
        -16 * sp.pi,
        R(15293365, 508032) + R(27145, 504) * n + R(3085, 72) * n**2,
        sp.pi * (R(38645, 756) - R(65, 9) * n) * (1 + 3 * sp.log(v)),
        R(11583231236531, 4694215680)
        - R(640, 3) * sp.pi**2
        - R(6848, 21) * (np.euler_gamma + sp.log(4 * v))
        + (-R(15737765635, 3048192) + R(2255, 12) * sp.pi**2) * n
        + R(76055, 1728) * n**2
        - R(127825, 1296) * n**3,
        sp.pi * (R(77096675, 254016) + R(378515, 1512) * n - R(74045, 756) * n**2),
    ]
    Psi = R(3, 128) / n * v**-5 * sum(c * v**i for i, c in enumerate(p))
    return sp.lambdify((F, n), -sp.diff(Psi, F), "numpy")


def tf2_dphi(f, eta):
    """phi'_TF2(f)"""
    return _tf2()(f, eta)


# --------------------------------------------------------------------------------------------------
# Step 1: single waveform
#
# Three frequency regions, bounded by fractions of the ringdown frequency f_RD.
# Phase derivative:
#     inspiral      eta phi' = eta phi'_TF2 + s1 + s2 f^(1/3) + s3 f^(2/3) + s4 f      (TaylorF2 3.5PN + pseudo-PN)
#     intermediate  eta phi' = b0 + b1/f + b2/f^2 + b3/f^3 + b4/f^4
#     merger-RD     eta phi' = c0 + c1/f + c2/f^2 + c4/f^4 + a / (f_damp^2 + (f - f_RD)^2)
# Amplitude, normalised by A0 = sqrt(2 eta / (3 pi^(1/3))) f^(-7/6):
#     inspiral      1 + a2 v^2 + a3 v^3 + a4 v^4 + a5 v^5,   v = (pi f)^(1/3)
#     intermediate  d0 + d1 f + d2 f^2 + d3 f^3
#     merger-RD     aR f_damp s / ((f - f_RD)^2 + (f_damp s)^2) * exp(-lam (f - f_RD) / (f_damp s))
# The regions are joined with smooth tanh windows (XAS imposes continuity instead).
# --------------------------------------------------------------------------------------------------
# Region boundaries, as fractions of f_RD: (inspiral | intermediate, intermediate | merger-ringdown)
CUTS = {"phase": (0.30, 0.80), "amp": (0.35, 0.85)}


def joins(kind, fRD):
    """The two frequencies where the regions of the phase (or of the amplitude) are joined."""
    c1, c2 = CUTS[kind]
    return c1 * fRD, c2 * fRD


def amp0(f, eta):
    """Leading-order amplitude A0"""
    return np.sqrt(2 * eta / (3 * np.pi ** (1 / 3))) * f ** (-7 / 6)


def nodes(lo, hi, n):
    """Chebyshev nodes (of the first kind) mapped to [lo, hi], extremes included."""
    k = np.arange(n)
    x = np.cos((2 * k + 1) * np.pi / (2 * n))[::-1]
    x = (x - x[0]) / (x[-1] - x[0])
    return lo + (hi - lo) * x


def win(f, f0, w):
    return 0.5 * (1 + np.tanh((f - f0) / w))


def blend(f, x1, x2, y_ins, y_int, y_mr):
    """Join the three regions at x1 and x2 with tanh windows."""
    w1, w2 = win(f, x1, 0.04 * x1), win(f, x2, 0.03 * x2)
    return (1 - w1) * y_ins + w1 * (1 - w2) * y_int + w2 * y_mr


class Region(object):
    """
    One frequency region of the phase derivative or of the amplitude. Parent class: the regions that are
    linear in their coefficients only define basis, target and model.
    """

    kind = None  # "phase" or "amp"
    piece = None  # 0: inspiral, 1: intermediate, 2: merger-ringdown

    def basis(self, f, fRD, fdamp):
        """Basis functions, as columns"""
        raise NotImplementedError("basis not implemented.")

    def target(self, f, eta, A, dphi):
        """What is fitted, from the data A(f), phi'(f)"""
        raise NotImplementedError("target not implemented.")

    def model(self, f, eta, y):
        """Back from the fitted quantity y to eta-independent phi' or A / A0"""
        raise NotImplementedError("model not implemented.")

    def fit_range(self, fRD, fdamp):
        """Frequency range of the fit (overlapping the neighbours, for the smooth joins)"""
        x1, x2 = joins(self.kind, fRD)
        return ((FMIN, 1.15 * x1), (0.85 * x1, 1.1 * x2), (0.9 * x2, fRD + 4 * fdamp))[
            self.piece
        ]

    def nodes(self, fRD, fdamp, n):
        """Collocation nodes"""
        return nodes(*self.fit_range(fRD, fdamp), n)

    def evaluate(self, f, coef, fRD, fdamp):
        return self.basis(f, fRD, fdamp) @ coef

    def fit(self, f, y, fRD, fdamp):
        """Coefficients from the data (f, y) in the fit range: linear least squares"""
        return np.linalg.lstsq(self.basis(f, fRD, fdamp), y, rcond=None)[0]

    def to_values(self, coef, fRD, fdamp):
        """What is fitted across parameter space (step 2): the values of the fit at the collocation nodes"""
        return self.evaluate(self.nodes(fRD, fdamp, coef.size), coef, fRD, fdamp)

    def from_values(self, vals, fRD, fdamp):
        """Coefficients from the values at the collocation nodes"""
        return np.linalg.solve(
            self.basis(self.nodes(fRD, fdamp, vals.size), fRD, fdamp), vals
        )


class PhaseInspiral(Region):
    """eta phi' = eta phi'_TF2 + s1 + s2 f^(1/3) + s3 f^(2/3) + s4 f"""

    kind, piece = "phase", 0

    def basis(self, f, fRD, fdamp):
        return np.stack([np.ones_like(f), f ** (1 / 3), f ** (2 / 3), f], 1)

    def target(self, f, eta, A, dphi):
        return eta * (dphi - tf2_dphi(f, eta))

    def model(self, f, eta, y):
        return tf2_dphi(f, eta) + y / eta


class PhaseIntermediate(Region):
    """eta phi' = b0 + b1/f + b2/f^2 + b3/f^3 + b4/f^4"""

    kind, piece = "phase", 1

    def basis(self, f, fRD, fdamp):
        return np.stack([np.ones_like(f), 1 / f, 1 / f**2, 1 / f**3, 1 / f**4], 1)

    def target(self, f, eta, A, dphi):
        return eta * dphi

    def model(self, f, eta, y):
        return y / eta


class PhaseMergerRingdown(PhaseIntermediate):
    """eta phi' = c0 + c1/f + c2/f^2 + c4/f^4 + a / (f_damp^2 + (f - f_RD)^2)"""

    piece = 2

    def basis(self, f, fRD, fdamp):
        lorentzian = 1 / (fdamp**2 + (f - fRD) ** 2)
        return np.stack([np.ones_like(f), 1 / f, 1 / f**2, 1 / f**4, lorentzian], 1)


class AmpInspiral(Region):
    """A / A0 = 1 + a2 v^2 + a3 v^3 + a4 v^4 + a5 v^5,   v = (pi f)^(1/3)"""

    kind, piece = "amp", 0

    def basis(self, f, fRD, fdamp):
        v = (np.pi * f) ** (1 / 3)
        return np.stack([v**2, v**3, v**4, v**5], 1)

    def target(self, f, eta, A, dphi):
        return A / amp0(f, eta) - 1

    def model(self, f, eta, y):
        return 1 + y


class AmpIntermediate(Region):
    """A / A0 = d0 + d1 f + d2 f^2 + d3 f^3"""

    kind, piece = "amp", 1

    def basis(self, f, fRD, fdamp):
        return np.stack([np.ones_like(f), f, f**2, f**3], 1)

    def target(self, f, eta, A, dphi):
        return A / amp0(f, eta)

    def model(self, f, eta, y):
        return y


class AmpMergerRingdown(AmpIntermediate):
    """
    A / A0 = aR f_damp s / ((f - f_RD)^2 + (f_damp s)^2) * exp(-lam (f - f_RD) / (f_damp s))
    Nonlinear in s and lam: nonlinear fit, and (aR, s, lam) are fitted directly across parameter space.
    """

    piece = 2

    def evaluate(self, f, coef, fRD, fdamp):
        aR, s, lam = coef
        return (
            aR
            * fdamp
            * s
            / ((f - fRD) ** 2 + (fdamp * s) ** 2)
            * np.exp(-lam * (f - fRD) / (fdamp * s))
        )

    def fit(self, f, y, fRD, fdamp):
        ansatz = lambda ff, aR, s, lam: self.evaluate(ff, (aR, s, lam), fRD, fdamp)
        return curve_fit(ansatz, f, y, p0=[0.5, 1.0, 0.5], maxfev=20000)[0]

    def to_values(self, coef, fRD, fdamp):
        return coef

    def from_values(self, vals, fRD, fdamp):
        return vals


REGIONS = {
    "ins": PhaseInspiral(),
    "int": PhaseIntermediate(),
    "mr": PhaseMergerRingdown(),
    "ains": AmpInspiral(),
    "aint": AmpIntermediate(),
    "amr": AmpMergerRingdown(),
}


class Ansatz:
    """The model for one binary: eta, f_RD, f_damp and the coefficients of each region."""

    def __init__(self, eta, fRD, fdamp, coef):
        self.eta, self.fRD, self.fdamp, self.coef = eta, fRD, fdamp, coef

    def __call__(self, f):
        """Amplitude and phi' on the frequencies f"""
        y = {
            name: reg.model(
                f, self.eta, reg.evaluate(f, self.coef[name], self.fRD, self.fdamp)
            )
            for name, reg in REGIONS.items()
        }
        dphi = blend(f, *joins("phase", self.fRD), y["ins"], y["int"], y["mr"])
        A = amp0(f, self.eta) * blend(
            f, *joins("amp", self.fRD), y["ains"], y["aint"], y["amr"]
        )
        return A, dphi


class Fit:
    """
    Step 1: fit the three regions to one waveform, on a log grid in frequency.
    ansatz: the fitted model; values: what is fitted across parameter space in step 2.
    """

    def __init__(self, q):
        self.q = q
        self.wf = wf = Waveform(q)
        self.fg = fg = np.geomspace(FMIN, FMAX, 1500)
        self.Ad, self.Pd = np.interp(fg, wf.f, np.abs(wf.H)), np.interp(
            fg, wf.f, wf.dphi
        )
        coef, self.values = {}, {}
        for name, reg in REGIONS.items():
            lo, hi = reg.fit_range(wf.fRD, wf.fdamp)
            k = (fg >= lo) & (fg <= hi)
            y = reg.target(fg, wf.eta, self.Ad, self.Pd)
            coef[name] = reg.fit(fg[k], y[k], wf.fRD, wf.fdamp)
            self.values[name] = reg.to_values(coef[name], wf.fRD, wf.fdamp)
        self.ansatz = Ansatz(wf.eta, wf.fRD, wf.fdamp, coef)


# --------------------------------------------------------------------------------------------------
# Step 2: across parameter space
#
# As in XAS, the linear coefficients are not fitted directly: we fit the values of each region's fit at a few
# Chebyshev nodes ("collocation points"), as polynomials in eta, and solve back for the coefficients.
# The nonlinear ringdown parameters (lam, s) and f_RD, f_damp are fitted directly in eta.
# --------------------------------------------------------------------------------------------------
class Phenom:
    def __init__(self, qs, deg=DEG):
        """Step 2: node values, f_RD, f_damp and the nonlinear ringdown parameters as polynomials in eta."""
        self.fits = [Fit(q) for q in qs]
        self.etas = np.array([fit.wf.eta for fit in self.fits])
        self.poly = {
            "fRD": np.polyfit(self.etas, [fit.wf.fRD for fit in self.fits], deg),
            "fdamp": np.polyfit(self.etas, [fit.wf.fdamp for fit in self.fits], deg),
        }
        for name in REGIONS:
            V = np.array([fit.values[name] for fit in self.fits])
            self.poly[name] = [
                np.polyfit(self.etas, V[:, j], deg) for j in range(V.shape[1])
            ]

    def ansatz(self, eta):
        """The model at any eta: evaluate the fits and solve back for the coefficients (collocation)."""
        fRD, fdamp = np.polyval(self.poly["fRD"], eta), np.polyval(
            self.poly["fdamp"], eta
        )
        coef = {}
        for name, reg in REGIONS.items():
            vals = np.array([np.polyval(c, eta) for c in self.poly[name]])
            coef[name] = reg.from_values(vals, fRD, fdamp)
        return Ansatz(eta, fRD, fdamp, coef)

    def __call__(self, q, f):
        """Amplitude and phi' of the model at mass ratio q, on the frequencies f"""
        return self.ansatz(eta_of(q))(f)


# --------------------------------------------------------------------------------------------------
# Step 3: comparison
#
# The model is evaluated at a mass ratio that is not in the training set and compared with EOB, in the
# frequency domain and, after an inverse FFT, in the time domain.
# --------------------------------------------------------------------------------------------------
def complex_fd(fg, A, dphi):
    """h~(f) = A e^{i Psi}, with Psi the integral of phi' (trapezoid on the grid)."""
    return A * np.exp(1j * cumulative_trapezoid(dphi, fg, initial=0.0))


def to_uniform(fg, Hg, f):
    """Interpolate amplitude and (unwrapped) phase of a complex FD waveform onto the FFT grid f."""
    A = np.interp(f, fg, np.abs(Hg), left=0, right=0)
    ph = np.interp(f, fg, np.unwrap(np.angle(Hg)))
    return A * np.exp(1j * ph)


def band_td(f, H, n, df):
    """Back to the time domain from the positive frequencies in [FMIN, FMAX] (smooth band edges)."""
    w = win(f, FMIN * 1.3, 0.1 * FMIN) * (1 - win(f, 0.9 * FMAX, 0.02 * FMAX))
    z = np.zeros(n, complex)
    i0 = int(round(f[0] / df))
    z[i0 : i0 + f.size] = H * w
    return np.fft.ifft(z) * n * df


class Comparison:
    """
    Model against an EOB waveform wf. Frequency domain: flat-noise match in [FMIN, FMAX], maximised over
    time and phase shifts. Time domain (time_domain()): inverse FFT of both, band-limited in the same way,
    the model aligned in time and phase.
    """

    def __init__(self, model, wf):
        self.wf = wf
        self.ansatz = model.ansatz(wf.eta)
        self.fg = np.geomspace(FMIN, FMAX, 3000)
        self.Am, self.Pm = self.ansatz(self.fg)
        kb = (wf.f >= FMIN) & (wf.f <= FMAX)
        self.fb, self.Hd = wf.f[kb], wf.H[kb]
        self.Hm = to_uniform(self.fg, complex_fd(self.fg, self.Am, self.Pm), self.fb)
        self.match, self.kshift, self.cmax = match(self.Hd, self.Hm, NSHIFT)
        self.mismatch = 1 - self.match

    def time_domain(self):
        """tt (peak of EOB at 0), hd, hm, and the phase difference dph (zero at t = -2500 M)"""
        fb = self.fb
        df = fb[1] - fb[0]
        n = 1 << int(np.ceil(np.log2(1 / (df * self.wf.dt))))
        hd = band_td(fb, self.Hd, n, df)
        hm = band_td(fb, align(self.Hm, self.kshift, self.cmax, NSHIFT), n, df)
        tt = np.arange(n) / (n * df)
        tt = np.where(
            tt > tt[-1] / 2, tt - tt[-1] - 1 / (n * df), tt
        )  # second half: negative times
        tt, hd, hm = (
            np.roll(x, int((tt < 0).sum())) for x in (tt, hd, hm)
        )  # in order of time
        self.tt = tt - tt[np.argmax(np.abs(hd))]
        self.hd, self.hm = hd, hm
        dph = np.unwrap(np.angle(hm)) - np.unwrap(np.angle(hd))
        self.dph = dph - dph[self.at(-2500)]
        return self

    def at(self, x):
        """Index of the time x"""
        return np.argmin(np.abs(self.tt - x))

    def amp_ratio(self, x):
        i = self.at(x)
        return np.abs(self.hm[i]) / np.abs(self.hd[i])


# --------------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------------
def fig_regions(fit):
    """Step 1 on one training waveform"""
    fg, Ad, Pd, fRD = fit.fg, fit.Ad, fit.Pd, fit.wf.fRD
    A, dphi = fit.ansatz(fg)
    f1, f2 = joins("phase", fRD)
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.6, 5.6))
    for ax in (a1, a2):
        ax.axvspan(FMIN, f1, color=C[0], alpha=0.08, lw=0)
        ax.axvspan(f1, f2, color=C[2], alpha=0.10, lw=0)
        ax.axvspan(f2, FMAX, color=C[1], alpha=0.08, lw=0)
    a1.loglog(fg, Ad, color=INK, lw=2.6, label=f"EOB, $q={fit.q:g}$")
    a1.loglog(fg, A, label="fit", **DASH)
    a1.set(
        xlabel=r"$Mf$",
        ylabel=r"$|\tilde h_{22}(f)|$",
        xlim=(FMIN, FMAX),
        ylim=(1e-2 * Ad.max() * 0.1, 3 * Ad.max()),
    )
    a1.legend(loc="lower left")
    for x, lab in (
        (np.sqrt(FMIN * f1), "inspiral"),
        (np.sqrt(f1 * f2), "intermediate"),
        (np.sqrt(f2 * FMAX), "merger-\nringdown"),
    ):
        a1.text(x, 2.2 * Ad.max(), lab, ha="center", va="top", fontsize=10.5, color=INK)
    k = fg > 0.02
    a2.plot(fg[k], Pd[k], color=INK, lw=2.6)
    a2.plot(fg[k], dphi[k], **DASH)
    a2.axvline(fRD, color=MUTED, lw=0.8)
    a2.text(fRD * 1.03, 0.85 * Pd[k].max(), r"$f_{\rm RD}$", color=INK, fontsize=11)
    a2.set(
        xlabel=r"$Mf$ (zoom on the late part)",
        ylabel=r"$\varphi'(f)=-2\pi\,t(f)$",
        xlim=(0.02, 0.14),
    )
    done(fig, "toy_phenom_regions")


def fig_coeffs(model):
    """Step 2: few node vals vs eta"""
    ee = np.linspace(0.08, 0.25, 200)
    et = eta_of(Q_TEST)
    fig, axes = plt.subplots(3, 1, figsize=(4.8, 5.6), sharex=True)
    show = [
        ("ins", 2, r"inspiral, value at node 3"),
        ("int", 2, r"intermediate, value at node 3"),
        ("mr", 4, r"merger-ringdown, value at node 5"),
    ]
    for ax, (name, j, lab) in zip(axes, show):
        poly = model.poly[name][j]
        vals = np.array([fit.values[name][j] for fit in model.fits])
        ax.plot(ee, np.polyval(poly, ee), color=C[0], lw=1.6)
        ax.plot(model.etas, vals, "o", color=INK, ms=6, label="training set", **MARK)
        ax.plot(
            et,
            np.polyval(poly, et),
            "D",
            color=C[1],
            ms=7,
            label=f"$q={Q_TEST:g}$ (test)",
            **MARK,
        )
        ax.set_title(lab, fontsize=11, loc="left", color=INK, pad=3)
        ax.set_ylabel(
            r"$\eta\,(\varphi'-\varphi'_{\rm TF2})$"
            if name == "ins"
            else r"$\eta\,\varphi'$"
        )
    axes[0].legend(loc="best", fontsize=9.5)
    axes[-1].set_xlabel(r"$\eta$")
    done(fig, "toy_phenom_coeffs")


def fig_fd(c):
    """Step 3: test q, frequency domain"""
    f, Pd, fb, fg, Pm, fRD = c.wf.f, c.wf.dphi, c.fb, c.fg, c.Pm, c.ansatz.fRD
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5.6, 5.4))
    a1.set_xlabel(r"$Mf$")
    a1.loglog(
        fb,
        np.abs(c.Hd),
        color=INK,
        lw=2.6,
        label=f"EOB, $q={c.wf.q:g}$ (not in the training set)",
    )
    a1.loglog(fg, c.Am, label="mock Phenom", **DASH)
    a1.set(ylabel=r"$|\tilde h_{22}(f)|$", xlim=(FMIN, FMAX))
    a1.legend(loc="lower left", fontsize=10)
    k, kb = fg > 0.02, fb > 0.02
    a2.plot(fb[kb], np.interp(fb[kb], f, Pd), color=INK, lw=2.6)
    a2.plot(fg[k], Pm[k] + (np.interp(fRD, f, Pd) - np.interp(fRD, fg, Pm)), **DASH)
    a2.set(
        xlabel=r"$Mf$ (zoom on the late part)",
        ylabel=r"$\varphi'(f)$",
        xlim=(0.02, 0.14),
    )
    done(fig, "toy_phenom_fd")


def fig_td(c):
    """Step 3: test q, time domain"""
    fig, (a1, a2) = zoom_td(
        c.tt, np.real(c.hd), np.real(c.hm), "EOB", "mock Phenom", (6.4, 4.8)
    )
    a1.set_ylim(-0.27, 0.32)
    a1.legend(loc="upper left", fontsize=10, ncol=2)
    done(fig, "toy_phenom_td")


def figures():
    model = Phenom(Q_TRAIN)
    fig_regions(model.fits[2])  # q = 2
    fig_coeffs(model)

    c = Comparison(model, Waveform(Q_TEST))
    print(
        f"test q = {Q_TEST:g}: f_RD model {c.ansatz.fRD:.5f} vs {c.wf.fRD:.5f}; "
        f"f_damp model {c.ansatz.fdamp:.5f} vs {c.wf.fdamp:.5f}"
    )
    print(f"flat-noise mismatch, Mf in [{FMIN}, {FMAX}]: {c.mismatch:.2e}")
    for fit in model.fits:  # sanity: refit = training waveform
        print(
            f"   training q = {fit.q:g}: mismatch {Comparison(model, fit.wf).mismatch:.2e}"
        )
    fig_fd(c)

    c.time_domain()
    fig_td(c)
    for tq in (-2000, -500, -100, 0):
        print(
            f"   phase difference at t = {tq:5d}: {c.dph[c.at(tq)]:+.3f} rad,"
            f"  amplitude ratio {c.amp_ratio(tq):.3f}"
        )


# --------------------------------------------------------------------------------------------------
# Outside the training set
# --------------------------------------------------------------------------------------------------
def extrapolation():
    """Model evaluated inside and outside the training range 1 <= q <= 8. Figures: toy_phenom_extrap, toy_phenom_q20."""
    missing = [q for q in Q_EXTRA if not cache(q).exists()]
    if missing:
        Waveform.generate(missing)
    model = Phenom(Q_TRAIN)
    dphi_peak, dphi_100 = [], []
    for q in Q_EXTRA:
        c = Comparison(model, Waveform(q)).time_domain()
        dphi_peak.append(c.dph[c.at(0)])
        dphi_100.append(c.dph[c.at(-100)])
        print(
            "q = %5.2f: mismatch %.1e, dphi(-100M) %+.3f, dphi(peak) %+.3f rad, A ratio(peak) %.3f, f_RD error %+.2e"
            % (
                q,
                c.mismatch,
                dphi_100[-1],
                dphi_peak[-1],
                c.amp_ratio(0),
                c.ansatz.fRD / c.wf.fRD - 1,
            )
        )
    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    ax.axvspan(Q_TRAIN[0], Q_TRAIN[-1], color=MUTED, alpha=0.15, lw=0)
    ax.text(
        2.8,
        0.75,
        "training range",
        color=INK,
        fontsize=10,
        transform=ax.get_xaxis_transform(),
    )
    ax.plot(
        Q_EXTRA, np.abs(dphi_peak), "o-", color=C[1], ms=7, label="at the peak", **MARK
    )
    ax.plot(
        Q_EXTRA,
        np.abs(dphi_100),
        "s--",
        color=C[0],
        ms=6,
        label=r"$100M$ before the peak",
        **MARK,
    )
    for xt in Q_TRAIN:
        ax.axvline(xt, color=MUTED, lw=0.6, ymax=0.06)
    ax.set(
        xscale="log", yscale="log", xlabel="$q$", ylabel=r"$|\Delta\phi_{22}|$ [rad]"
    )
    ax.set_xticks([1, 2, 5, 10, 20])
    ax.set_xticklabels(["1", "2", "5", "10", "20"])
    ax.legend(loc="upper left")
    done(fig, "toy_phenom_extrap")

    # one example far outside the training range: the last one, q = 20
    tt, hd = c.tt, np.real(c.hd)
    fig, (a1, a2) = zoom_td(
        tt, hd, np.real(c.hm), "EOB, $q=20$", "mock Phenom (extrapolated)", (5.4, 3.9)
    )
    lim = 1.25 * np.abs(hd[(tt > -1200) & (tt < 60)]).max()
    a1.set_ylim(-lim, lim)
    done(fig, "toy_phenom_q20")


if __name__ == "__main__":
    if sys.argv[1:] == ["data"]:
        Waveform.generate()
    elif sys.argv[1:] == ["extrap"]:
        extrapolation()
    else:
        figures()
