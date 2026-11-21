"""
Radiation reaction: the force added to dp/dt (per unit mu), in Cartesian components.
"""

import numpy as np


class Flux(object):
    """
    Radiation-reaction force.
    Parent class.
    """

    def __init__(self):
        self.nu = None
        self.flux_type = None

    def F(self, x, p, v):
        raise NotImplementedError("F Method not implemented.")


class NoFlux(Flux):
    """
    Conservative dynamics.
    """

    def __init__(self, nu=0.0):
        self.nu = nu
        self.flux_type = "cons"

    def F(self, x, p, v):
        return np.zeros(3)


class QuadrupoleFlux(Flux):
    """
    Leading-order (2.5PN, quadrupolar) radiation reaction in harmonic gauge,

        F = nu (8 / 5 r^3) [ (3 v^2 + 17/(3r)) rdot n - (v^2 + 3/r) v ]
    """

    def __init__(self, nu):
        self.nu = nu
        self.flux_type = "diss-quadrupole"

    def F(self, x, p, v):
        r = np.sqrt(x @ x)
        n = x / r
        rdot, v2 = n @ v, v @ v
        return (
            self.nu
            * 8
            / (5 * r**3)
            * ((3 * v2 + 17 / (3 * r)) * rdot * n - (v2 + 3 / r) * v)
        )
