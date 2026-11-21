"""
Hamiltonians for a point particle (G = c = M = mu = 1).

Cartesian-like coordinates x = r n, where (r, theta, phi) are Boyer-Lindquist and n is the unit
vector of (theta, phi); p is the conjugate momentum (per unit mu); a is the Kerr spin vector (any
direction):

    H = (2 r / Lambda) a.(x cross p)
        + sqrt{ (Delta Sigma / Lambda) [ 1 + (r^2/Sigma) ( p^2 + (Delta/r^2 - 1)(n.p)^2
                                         - (r^2 + 2r + (n.a)^2)/Lambda ((n cross p).a)^2 ) ] }
    Sigma = r^2 + (n.a)^2,  Delta = r^2 - 2r + a^2,  Lambda = (r^2 + a^2)^2 - Delta (a^2 - (n.a)^2)

The gradients of H are generated symbolically (sympy) and compiled once, at the first call.
"""

from functools import lru_cache

import numpy as np
from scipy.optimize import brentq


class Hamiltonian(object):
    """
    Parent class for all Hamiltonians.
    """

    def __init__(self):
        self.ham_type = None
        self.avec = None
        self.rH = None

    def ham(self, x, p):
        """
        Hamiltonian and its derivatives: H, dH/dx, dH/dp
        """
        raise NotImplementedError("Hamiltonian not implemented.")

    def energy(self, x, p):
        return self.ham(x, p)[0]

    def velocity(self, x, p):
        """dx/dt = dH/dp"""
        return self.ham(x, p)[2]


@lru_cache(maxsize=None)
def _kerr_functions():
    """H(x, p, a) and its gradient with respect to (x, p), from sympy."""
    import sympy as sp

    X = sp.symbols("x y z", real=True)
    P = sp.symbols("p_x p_y p_z", real=True)
    A = sp.symbols("a_x a_y a_z", real=True)
    x, p, a = sp.Matrix(X), sp.Matrix(P), sp.Matrix(A)
    r = sp.sqrt(x.dot(x))
    n = x / r
    a2, na = a.dot(a), n.dot(a)
    Sig, Del = r**2 + na**2, r**2 - 2 * r + a2
    Lam = (r**2 + a2) ** 2 - Del * (a2 - na**2)
    Q = 1 + r**2 / Sig * (
        p.dot(p)
        + (Del / r**2 - 1) * n.dot(p) ** 2
        - (r**2 + 2 * r + na**2) / Lam * (n.cross(p)).dot(a) ** 2
    )
    H = 2 * r / Lam * a.dot(x.cross(p)) + sp.sqrt(Del * Sig / Lam * Q)
    args = (*X, *P, *A)
    grad = [sp.diff(H, s) for s in (*X, *P)]
    return sp.lambdify(args, H, "numpy", cse=True), sp.lambdify(
        args, grad, "numpy", cse=True
    )


class Kerr(Hamiltonian):
    """
    Kerr Hamiltonian, generic orbits.
    a: Kerr parameter. A number (spin along z, negative = antialigned) or a 3-vector.
    """

    def __init__(self, a=0.0):
        super(Kerr, self).__init__()
        self.ham_type = "kerr"
        self.avec = np.array([0.0, 0.0, a]) if np.isscalar(a) else np.asarray(a, float)
        if self.avec @ self.avec >= 1:
            raise ValueError("Kerr parameter must be |a| < 1.")
        self.a = np.linalg.norm(self.avec)
        self.rH = 1 + np.sqrt(1 - self.a**2)

    def ham(self, x, p):
        """
        Hamiltonian and its derivatives: H, dH/dx, dH/dp.
        x, p: shape (3,), or (N, 3) for N points at once.
        """
        H, grad = _kerr_functions()
        x, p = np.asarray(x, float), np.asarray(p, float)
        args = (*x.T, *p.T, *self.avec)
        g = np.array(np.broadcast_arrays(*grad(*args)))
        return H(*args), g[:3].T, g[3:].T

    def constants(self, x, p):
        """
        Constants of geodesic motion (per unit mu): energy E, angular momentum along the spin
        L_z = L.a/|a| (along z if a = 0), and Carter constant
            Q = p_theta^2 + cos^2(theta) [ a^2 (1 - E^2) + L_z^2 / sin^2(theta) ]
              = L^2 - L_z^2 + (n.a)^2 (1 - E^2),        L = x cross p,
        with theta the Boyer-Lindquist polar angle from the spin axis.
        """
        x, p = np.asarray(x, float), np.asarray(p, float)
        E = self.energy(x, p)
        L = np.cross(x, p)
        zhat = self.avec / self.a if self.a > 0 else np.array([0.0, 0.0, 1.0])
        Lz = L @ zhat
        na = (x @ self.avec) / np.linalg.norm(x, axis=-1)
        Q = np.sum(L * L, axis=-1) - Lz**2 + na**2 * (1 - E**2)
        return E, Lz, Q

    # equatorial orbits
    def __az__(self):
        """Signed Kerr parameter, for the equatorial formulas (spin along z)."""
        if self.avec[0] or self.avec[1]:
            raise ValueError("Equatorial formulas need the spin along z.")
        return self.avec[2]

    def circ_j(self, r):
        """
        Angular momentum of the circular equatorial orbit (Bardeen, Press & Teukolsky 1972):
        prograde for a > 0, retrograde for a < 0
        """
        a = self.__az__()
        return (r**2 - 2 * a * np.sqrt(r) + a**2) / (
            r**0.75 * np.sqrt(r**1.5 - 3 * np.sqrt(r) + 2 * a)
        )

    def circ_omega(self, r):
        """Orbital frequency of the circular equatorial orbit"""
        return 1 / (r**1.5 + self.__az__())

    def r_isco(self):
        """Innermost stable circular orbit"""
        a = self.__az__()
        return brentq(
            lambda r: r**2 - 6 * r + 8 * a * np.sqrt(r) - 3 * a**2,
            1.0 + 1e-9 if a > 0 else 5.9,
            9.1,
        )

    def r_lightring(self):
        """Circular photon orbit"""
        return 2 * (1 + np.cos(2 / 3 * np.arccos(-self.__az__())))

    def qnm_220(self):
        """
        Fundamental l = m = 2 quasinormal mode, fit of Berti, Cardoso & Will (2006): (omega, tau).
        Valid for the corotating mode, 0 <= a < 1.
        """
        a = self.__az__()
        if a < 0:
            raise ValueError("qnm_220: fit valid for a >= 0 only.")
        om = 1.5251 - 1.1568 * (1 - a) ** 0.1292
        Q = 0.7000 + 1.4187 * (1 - a) ** (-0.4990)
        return om, 2 * Q / om


class Schwarzschild(Kerr):
    """
    Schwarzschild Hamiltonian
    """

    def __init__(self):
        super(Schwarzschild, self).__init__(0.0)
        self.ham_type = "schwarzschild"
