"""
Toy model 1 in a few lines: a particle on Kerr with quadrupolar radiation reaction.

    python example.py            # quasi-circular inspiral and plunge, spin aligned with the orbit
    python example.py precessing # inclined orbit around a black hole with a tilted spin
"""

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from style import C  # noqa: E402  (shared matplotlib style, scripts/style.py)

import hamiltonian
import flux as flx
import dynamics as dyn
import waveform

if __name__ == "__main__":

    nu = 0.05
    if "precessing" in sys.argv[1:]:
        ham = hamiltonian.Kerr([0.6, 0.0, 0.5])
        ics, ic_pars = "spherical", {"r0": 9.0, "iota": 0.5}
    else:
        ham = hamiltonian.Kerr(0.7)
        ics, ic_pars = "circular", {"r0": 1.6 * ham.r_isco()}

    # dynamics
    d = dyn.Dynamics(
        nu,
        ham=ham,
        flux=flx.QuadrupoleFlux(nu),
        ics=ics,
        ic_pars=ic_pars,
        dt=0.5,
        verbose=True,
    )
    d.evolve()

    # waveform
    wf = waveform.QuadrupoleWaveform(d)
    wf.compute_modes([(2, 2), (2, 1)])

    fig = plt.figure(figsize=(10, 4.5))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot(*d.x.T, lw=0.8, color=C[0])
    ax.set(xlabel="x", ylabel="y", zlabel="z", title="orbit")
    ax = fig.add_subplot(2, 2, 2)
    ax.plot(d.t, d.r, color=C[0])
    ax.set(ylabel="r")
    ax = fig.add_subplot(2, 2, 4)
    ax.plot(wf.t, np.real(wf.hlm[(2, 2)]), label="Re h22", color=C[0], lw=1.2)
    ax.plot(wf.t, np.real(wf.hlm[(2, 1)]), label="Re h21", color=C[1], lw=1.2)
    ax.set(xlabel="t", ylabel="strain")
    ax.legend()
    fig.tight_layout()
    plt.show()
