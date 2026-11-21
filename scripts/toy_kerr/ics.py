"""
Initial data (x0, p0) for the dynamics.
"""

import numpy as np
from scipy.optimize import brentq

import hamiltonian
import flux as flx


class ICs(object):
    """
    Initial conditions for a given Hamiltonian and flux. kind and ic_pars:

    'circular'  : r0, post_adiabatic (default True)   equatorial, quasi-circular
    'eccentric' : p, e, at ('apastron' or 'periastron')  equatorial geodesic, semilatus rectum and eccentricity
    'energy'    : E0, j0, r0                           equatorial, ingoing, from energy and angular momentum
    'spherical' : r0, iota                             inclined quasi-spherical orbit
    'cartesian' : x0, p0                               given directly

    The equatorial kinds need the spin along z.
    """

    def __init__(self, nu, ham=None, flux=None, kind="circular", ic_pars=None):
        self.nu = nu
        self.kind = kind
        self.ic_pars = {} if ic_pars is None else ic_pars

        if not isinstance(ham, hamiltonian.Hamiltonian):
            raise ValueError("Hamiltonian not recognized.")
        self.ham = ham

        if not isinstance(flux, flx.Flux):
            raise ValueError("Flux not recognized.")
        self.flux = flux

        kinds = {
            "circular": self.__circular_ic__,
            "eccentric": self.__eccentric_ic__,
            "energy": self.__energy_ic__,
            "spherical": self.__spherical_ic__,
            "cartesian": self.__cartesian_ic__,
        }
        if kind not in kinds:
            raise ValueError(
                "ICs kind not recognized: {}. Choose among {}".format(kind, list(kinds))
            )
        self.__ic__ = kinds[kind]

    def __initial_conditions__(self):
        """
        Initial state vector y0 = (x, y, z, px, py, pz)
        """
        x0, p0 = self.__ic__()
        return np.concatenate([x0, p0])

    def __circular_ic__(self):
        """
        Equatorial quasi-circular initial data at r0.
        post_adiabatic=False: exactly circular geodesic data (p_r = 0).
        post_adiabatic=True : p_r such that rdot = jdot / (dj_c/dr), the leading adiabatic inspiral rate.
        """
        r0 = self.ic_pars["r0"]
        j = self.ham.circ_j(r0)
        x0, p0 = np.array([r0, 0.0, 0.0]), np.array([0.0, j / r0, 0.0])
        if self.ic_pars.get("post_adiabatic", True) and self.flux.flux_type != "cons":
            v = self.ham.velocity(x0, p0)
            jdot = np.cross(x0, self.flux.F(x0, p0, v))[
                2
            ]  # torque on the circular orbit
            h = 1e-4 * r0
            djdr = (self.ham.circ_j(r0 + h) - self.ham.circ_j(r0 - h)) / (2 * h)
            rdot = jdot / djdr
            # dr/dt = dH/dp_r is linear in p_r for small p_r
            eps = 1e-6
            vr = self.ham.velocity(x0, p0 + np.array([eps, 0, 0]))[0]
            p0 = p0 + np.array([rdot / (vr / eps), 0, 0])
        return x0, p0

    def __eccentric_ic__(self):
        """
        Equatorial geodesic with semilatus rectum p and eccentricity e, started at apastron (or
        periastron): (E, j) from the two turning points r = p/(1 -+ e), where H(r, p_r = 0, j) = E.
        """
        p, e = self.ic_pars["p"], self.ic_pars["e"]
        at = self.ic_pars.get("at", "apastron")
        if at not in ("apastron", "periastron"):
            raise ValueError("at must be 'apastron' or 'periastron'.")
        ra, rp = p / (1 - e), p / (1 + e)
        Hof = lambda r, j: self.ham.energy(
            np.array([r, 0, 0.0]), np.array([0, j / r, 0.0])
        )
        jc = self.ham.circ_j(p)
        j = brentq(lambda j: Hof(ra, j) - Hof(rp, j), 0.5 * jc, 3 * jc)
        r0 = ra if at == "apastron" else rp
        return np.array([r0, 0.0, 0.0]), np.array([0.0, j / r0, 0.0])

    def __energy_ic__(self):
        """
        Equatorial initial data from (E0, j0, r0), ingoing: solve H(r0, p_r, j0) = E0 for p_r < 0.
        """
        E0, j0, r0 = self.ic_pars["E0"], self.ic_pars["j0"], self.ic_pars["r0"]
        self.ham.__az__()  # equatorial: the spin must be along z
        x0 = np.array([r0, 0.0, 0.0])
        f = lambda pr: self.ham.energy(x0, np.array([pr, j0 / r0, 0.0])) - E0
        if f(0.0) > 0:
            raise ValueError(
                "E0 is below the effective potential at r0: no solution for p_r."
            )
        return x0, np.array([brentq(f, -5.0, 0.0), j0 / r0, 0.0])

    def __spherical_ic__(self):
        """
        Inclined quasi-spherical orbit: start on the x axis with p in the (y, z) plane at inclination
        iota from the equatorial plane of the frame; |p| is fixed by d(n.p)/dt = 0 (no radial acceleration).
        """
        r0, iota = self.ic_pars["r0"], self.ic_pars["iota"]
        x0 = np.array([r0, 0.0, 0.0])
        d = np.array([0.0, np.cos(iota), np.sin(iota)])
        n = x0 / r0

        def prdot(L):
            p0 = L / r0 * d
            _, dHdx, v = self.ham.ham(x0, p0)
            return (v - n * (n @ v)) / r0 @ p0 - n @ dHdx

        L = brentq(prdot, 0.5 * np.sqrt(r0), 3 * np.sqrt(r0) + 3)
        return x0, L / r0 * d

    def __cartesian_ic__(self):
        return np.asarray(self.ic_pars["x0"], float), np.asarray(
            self.ic_pars["p0"], float
        )
