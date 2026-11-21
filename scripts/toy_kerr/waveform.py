"""
Waveforms from the dynamics.

Mass and current quadrupole formula (per unit nu M / D):
    h_ij = 2 [ d^2 I_ij/dt^2 + (4/3) eps_{ab(i} d^2 J_{j)a}/dt^2 N_b ]^TT,
    I_ij = STF x_i x_j,   J_ij = STF x_i (x cross v)_j,
decomposed on spin-weighted (s = -2), l = 2 harmonics by quadrature on the sphere.
"""

import numpy as np
from scipy.integrate import cumulative_trapezoid

from utils import EPS, stf, sYlm2, rot_z, rot_y


class Waveform(object):
    """
    From a dynamics object, compute the waveform.
    """

    def __init__(self, dyn):
        self.dyn = dyn
        self.hlm = {}
        self.t = dyn.t

    def compute_modes(self, modes):
        for mode in modes:
            l, m = mode
            self.compute_mode(l, m)
        return self.hlm

    def compute_mode(self, l, m):
        """
        Compute the mode (l, m) of the waveform.
        This is a placeholder method and should be implemented in subclasses.
        """
        raise NotImplementedError(
            "compute_mode method not implemented for mode ({}, {}).".format(l, m)
        )


class QuadrupoleWaveform(Waveform):
    """
    Compute the quadrupolar waveform: mass quadrupole, plus the current quadrupole if current=True.
    Only l = 2 modes.
    """

    def __init__(self, dyn, current=True):
        super().__init__(dyn)
        self.waveform_type = "Quadrupole"
        self.current = current
        self.moments = self.radiative_moments()

    def radiative_moments(self):
        """
        Second time derivatives of the mass and current quadrupoles along the orbit (numerical d/dt).
        """
        t, x, v = self.dyn.t, self.dyn.x, self.dyn.v
        d = lambda f: np.gradient(f, t, axis=0, edge_order=2)
        I2 = d(d(stf(x[:, :, None] * x[:, None, :])))
        if self.current:
            J2 = d(d(stf(x[:, :, None] * np.cross(x, v)[:, None, :])))
        else:
            J2 = np.zeros_like(I2)
        return I2, J2

    def strain(self, theta, phi, moments=None):
        """
        h_+ - i h_x (in units of nu M / D) for an observer in the direction (theta, phi).
        moments: (I2, J2) to use instead of those of the orbit (e.g. rotated to another frame).
        """
        I2, J2 = self.moments if moments is None else moments
        N = np.array(
            [np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)]
        )
        eth = np.array(
            [np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)]
        )
        eph = np.array([-np.sin(phi), np.cos(phi), 0.0])
        K = np.einsum("abi,tja,b->tij", EPS, J2, N)  # eps_{ab i} J_{j a} N_b
        h = 2 * (I2 + 4 / 3 * 0.5 * (K + np.swapaxes(K, 1, 2)))
        hp = 0.5 * (
            np.einsum("i,tij,j->t", eth, h, eth) - np.einsum("i,tij,j->t", eph, h, eph)
        )
        hx = np.einsum("i,tij,j->t", eth, h, eph)
        return hp - 1j * hx

    def __project__(self, moments=None):
        """
        l = 2 spin-weighted modes h_2m(t), m = -2..2, by Gauss-Legendre x uniform quadrature on the sphere.
        """
        mu_, w = np.polynomial.legendre.leggauss(8)
        phis = np.arange(12) * 2 * np.pi / 12
        out = {m: 0 for m in range(-2, 3)}
        for c, wi in zip(mu_, w):
            th = np.arccos(c)
            for ph in phis:
                h = self.strain(th, ph, moments=moments)
                for m in out:
                    out[m] = out[m] + wi * (2 * np.pi / 12) * h * np.conj(
                        sYlm2(m, th, ph)
                    )
        return out

    def compute_mode(self, l, m):
        """
        Compute the mode (l, m) of the waveform, in the inertial frame of the dynamics.
        """
        if l != 2:
            raise NotImplementedError(
                "QuadrupoleWaveform: only l = 2 modes, not ({}, {}).".format(l, m)
            )
        if (2, m) not in self.hlm:
            for mm, h in self.__project__().items():
                self.hlm[(2, mm)] = h
        return self.hlm[(2, m)]

    def coprecessing(self):
        """
        Co-precessing and co-orbital l = 2 modes.

        Co-precessing frame: z' along L_N = x cross v; rotation R = Rz(alpha) Ry(beta) Rz(gamma) with the
        minimal-rotation condition dgamma/dt = -dalpha/dt cos(beta) (Boyle, Owen & Pfeiffer 2011). The
        radiative moments are rotated into that frame (T' = R^T T R) and decomposed there. Co-orbital: also
        remove the orbital phase Phi measured in the co-precessing frame, h_2m e^{i m Phi}.
        Returns dict(cp={(2, m): h_2m}, co={(2, m): h_2m}, R=(N,3,3), Lhat=(N,3), Phi=(N,)).
        """
        t = self.dyn.t
        Ln = np.cross(self.dyn.x, self.dyn.v)
        Ln = Ln / np.linalg.norm(Ln, axis=1)[:, None]
        al = np.unwrap(np.arctan2(Ln[:, 1], Ln[:, 0]))
        be = np.arccos(np.clip(Ln[:, 2], -1, 1))
        ga = -cumulative_trapezoid(np.gradient(al, t) * np.cos(be), t, initial=0)
        R = rot_z(al) @ rot_y(be) @ rot_z(ga)
        rot = lambda T: np.einsum("tai,tab,tbj->tij", R, T, R)
        I2, J2 = self.moments
        cp = {
            (2, m): h for m, h in self.__project__(moments=(rot(I2), rot(J2))).items()
        }
        xp = np.einsum("tai,ta->ti", R, self.dyn.x)
        Phi = np.unwrap(np.arctan2(xp[:, 1], xp[:, 0]))
        co = {lm: h * np.exp(1j * lm[1] * Phi) for lm, h in cp.items()}
        return dict(cp=cp, co=co, R=R, Lhat=Ln, Phi=Phi)


if __name__ == "__main__":
    # self-test: circular orbit at r = 10 on Schwarzschild, no radiation reaction
    import hamiltonian
    import dynamics as dyn

    ham = hamiltonian.Schwarzschild()
    d = dyn.Dynamics(
        0.0, ham=ham, ics="circular", ic_pars={"r0": 10.0}, tmax=600, dt=0.5
    )
    d.evolve()
    wf = QuadrupoleWaveform(d)
    wf.compute_modes([(2, m) for m in range(-2, 3)])
    Om = ham.circ_omega(10.0)
    amp = 8 * np.sqrt(np.pi / 5) * 10.0**2 * Om**2
    k = slice(50, -50)
    a22 = np.mean(np.abs(wf.hlm[(2, 2)][k]))
    w22 = np.mean(np.gradient(np.unwrap(np.angle(wf.hlm[(2, 2)][k])), wf.t[k]))
    assert abs(a22 / amp - 1) < 1e-3 and abs(w22 / (-2 * Om) - 1) < 1e-6
    print("|h22| / (8 sqrt(pi/5) r^2 Omega^2) =", a22 / amp)
    print("|h2-2| / |h22| =", np.mean(np.abs(wf.hlm[(2, -2)][k])) / a22)
    print("d arg(h22)/dt / (-2 Omega) =", w22 / (-2 * Om))
    print(
        "|h21|, |h20| relative:",
        np.mean(np.abs(wf.hlm[(2, 1)][k])) / amp,
        np.mean(np.abs(wf.hlm[(2, 0)][k])) / amp,
    )
