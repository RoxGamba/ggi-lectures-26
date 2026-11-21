"""
Utilities shared by the mock models (toy_phenom.py, toy_surrogate.py): the TEOBResumS waveforms that play
the role of NR simulations, and the match between two waveforms.
"""

import numpy as np


def eta_of(q):
    """Symmetric mass ratio"""
    return q / (1 + q) ** 2


def eob_22(q, f0):
    """
    Non-spinning TEOBResumS (2,2) mode from the initial GW frequency f0 (M f), geometric units:
    t, |h22|, phi22 on the native (non-uniform) time grid of the ODE solver, with h22 = A e^{-i phi}.

    Needs EOBRun_module (TEOBResumS). The built-in uniform interpolation returns NaN phases for some q
    when a single native sample is NaN: we ask for the native output and drop non-finite samples.
    The native grid is adaptive in the inspiral and has a step of 0.5 M through merger (the default
    dt_merger_interp of TEOBResumS).
    Returns t, A, phi and the number of dropped samples.
    """
    import EOBRun_module as E

    pars = {
        "M": 1.0,
        "q": q,
        "chi1": 0.0,
        "chi2": 0.0,
        "LambdaAl2": 0.0,
        "LambdaBl2": 0.0,
        "distance": 1.0,
        "initial_frequency": f0,
        "use_geometric_units": "yes",
        "interp_uniform_grid": "no",
        "domain": 0,
        "inclination": 0.0,
        "output_hpc": "no",
        "use_mode_lm": [1],
        "arg_out": "yes",
        "coalescence_angle": 0.0,
        "ode_tmax": 1e8,
    }
    out = E.EOBRunPy(pars)
    t, hlm = np.asarray(out[0]), out[3]
    A, ph = np.asarray(hlm["1"][0]), np.asarray(hlm["1"][1])
    ok = np.isfinite(A) & np.isfinite(ph)
    return t[ok], A[ok] * eta_of(q), ph[ok], int((~ok).sum())  # TEOB gives |h22| / nu


# match
def match(H1, H2, nshift=None):
    """
    Flat-noise match between two spectra on the same uniform frequency grid, maximised over a time and
    a phase shift (FFT over the time shift, resolution 1 / (nshift df)).
    Returns the match, the index k of the best shift and the complex correlation c[k] (for align).
    """
    nshift = H1.size if nshift is None else nshift
    assert H1.size <= nshift, "more frequency samples than nshift: increase it"
    z = np.zeros(nshift, complex)
    z[: H1.size] = H1 * np.conj(H2)
    c = np.fft.ifft(z) * nshift  # c[k] = sum_j H1_j conj(H2_j) e^{+2 pi i j k / nshift}
    nrm = np.sqrt(np.sum(np.abs(H1) ** 2) * np.sum(np.abs(H2) ** 2))
    k = np.argmax(np.abs(c))
    return np.abs(c[k]) / nrm, k, c[k]


def align(H2, k, ck, nshift=None):
    """H2 shifted in time and phase by the maximum found by match(H1, H2): <H1|align(H2)> = |c[k]|."""
    nshift = H2.size if nshift is None else nshift
    return (
        H2
        * np.exp(-2j * np.pi * np.arange(H2.size) * k / nshift)
        * np.exp(1j * np.angle(ck))
    )


def mismatch_td(h1, h2):
    """Flat-noise mismatch between two complex time series on the same grid, maximised over a phase and
    a time shift (in steps of the sampling time)."""
    n = 1 << int(np.ceil(np.log2(h1.size)) + 1)
    return 1 - match(np.fft.fft(h1, n), np.fft.fft(h2, n))[0]
