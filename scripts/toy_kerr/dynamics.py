"""
Evolve the dynamics of a point particle: Hamilton's equations plus a radiation-reaction force,

    dx/dt = dH/dp,     dp/dt = -dH/dx + F(x, p, v)
"""

import numpy as np
from scipy.integrate import solve_ivp

import hamiltonian
import flux as flx
import ics as ic


class Dynamics(object):
    """
    Evolve the dynamics of a particle on Kerr spacetime (generic orbits, any spin direction).
    Solve the system in the Hamiltonian formalism, in Cartesian-like coordinates.

    The evolution stops at r < r_stop (default: 1.03 r_H), r > r_max (if given), or at the final time.
    The solution is stored on a uniform time grid of step dt:
    t, x, p, v (N, 3 arrays), r, L = x cross p, and the constants of geodesic motion E, Lz (angular
    momentum along the spin), Q (Carter constant): conserved when there is no radiation reaction.
    """

    def __init__(
        self,
        nu,
        ham=None,
        flux=None,
        ics="circular",
        ic_pars=None,
        tmax=1e5,
        dt=1.0,
        r_stop=None,
        r_max=None,
        ode_method="DOP853",
        rtol=1e-10,
        atol=1e-12,
        verbose=False,
    ):
        self.nu = nu
        self.ic_pars = ic_pars
        self.tmax = tmax
        self.dt = dt
        self.sol = None
        self.ode_method = ode_method
        self.rtol = rtol
        self.atol = atol

        if not isinstance(ham, hamiltonian.Hamiltonian):
            raise ValueError("Hamiltonian not recognized.")
        self.ham = ham

        if flux is None:
            flux = flx.QuadrupoleFlux(nu) if nu else flx.NoFlux(nu)
        if not isinstance(flux, flx.Flux):
            raise ValueError("Flux not recognized.")
        self.flux = flux

        self.r_stop = 1.03 * self.ham.rH if r_stop is None else r_stop
        self.r_max = r_max

        if verbose:
            print("nu \t= {}".format(self.nu))
            print("Ham \t= {}".format(self.ham.ham_type))
            print("a \t= {}".format(self.ham.avec))
            print("Flux \t= {}".format(self.flux.flux_type))
            print("ICs \t= {}".format(ics))
            print("IC pars = {}".format(ic_pars))
            print("Tmax \t= {}".format(tmax))
            print("ODE \t= {}".format(ode_method))

        # Integration var names and indexes maps
        self.VIN = {0: "x", 1: "y", 2: "z", 3: "px", 4: "py", 5: "pz"}
        self.VNI = {v: i for i, v in self.VIN.items()}
        self.nvar = len(self.VIN.keys())

        # ICs
        self.ics = ic.ICs(nu, ham=ham, flux=flux, kind=ics, ic_pars=ic_pars)

    def __solve__(self, y0, tspan):
        """
        Solve ODEs
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html#scipy.integrate.solve_ivp
        """
        events = [self.__dyn_stop__]
        if self.r_max is not None:
            events.append(self.__dyn_escape__)
        return solve_ivp(
            self.__rhs__,
            tspan,
            y0,
            method=self.ode_method,
            rtol=self.rtol,
            atol=self.atol,
            dense_output=True,
            events=events,
        )

    def __dyn_stop__(self, t, y):
        """
        Stopping condition for dynamics:
        Stop evolution just before horizon
        """
        return np.sqrt(y[:3] @ y[:3]) - self.r_stop

    __dyn_stop__.terminal = True
    __dyn_stop__.direction = -1

    def __dyn_escape__(self, t, y):
        """
        Stopping condition for dynamics:
        Stop evolution when the particle escapes beyond r_max
        """
        return np.sqrt(y[:3] @ y[:3]) - self.r_max

    __dyn_escape__.terminal = True
    __dyn_escape__.direction = 1

    def __update_sol__(self, sol):
        """
        Store the solution on a uniform time grid, with a few derived quantities.
        """
        self.sol = sol
        self.t = np.arange(sol.t[0], sol.t[-1], self.dt)
        y = sol.sol(self.t)
        self.x, self.p = y[:3].T, y[3:].T
        self.v = self.ham.velocity(self.x, self.p)
        self.r = np.linalg.norm(self.x, axis=1)
        self.L = np.cross(self.x, self.p)
        self.E, self.Lz, self.Q = self.ham.constants(self.x, self.p)

    def evolve(self, tspan=None):
        """
        Solve the ODEs from the initial conditions. tspan: [t0, t1] (default: [0, tmax]).
        """
        tspan = [0, self.tmax] if tspan is None else tspan
        y0 = self.ics.__initial_conditions__()
        self.__update_sol__(self.__solve__(y0, tspan))
        return self.sol

    def __rhs__(self, t, y):
        """
        Compute the right-hand side of the ODE.
        """
        x, p = y[:3], y[3:]
        _, dHdx, dHdp = self.ham.ham(x, p)
        return np.concatenate([dHdp, -dHdx + self.flux.F(x, p, dHdp)])

    def phi(self):
        """Azimuthal angle of the orbit in the (x, y) plane, unwrapped"""
        return np.unwrap(np.arctan2(self.x[:, 1], self.x[:, 0]))


if __name__ == "__main__":
    # self-tests
    # 1. circular geodesic on Schwarzschild: r and E conserved
    dyn = Dynamics(
        0.0,
        ham=hamiltonian.Schwarzschild(),
        ics="circular",
        ic_pars={"r0": 10.0},
        tmax=600,
        dt=0.5,
    )
    dyn.evolve()
    assert np.ptp(dyn.r) < 1e-7 and np.ptp(dyn.E) < 1e-10
    print(
        "Schwarzschild, r = 10: r drift {:.1e}, E drift {:.1e}".format(
            np.ptp(dyn.r), np.ptp(dyn.E)
        )
    )

    # 2. circular geodesic on Kerr: orbital frequency against Bardeen, Press & Teukolsky
    ham = hamiltonian.Kerr(0.7)
    dyn = Dynamics(0.0, ham=ham, ics="circular", ic_pars={"r0": 6.0}, tmax=300, dt=0.5)
    dyn.evolve()
    ph = dyn.phi()
    Om = (ph[-1] - ph[0]) / (dyn.t[-1] - dyn.t[0])
    assert np.ptp(dyn.r) < 1e-7 and abs(Om / ham.circ_omega(6.0) - 1) < 1e-8
    print(
        "Kerr a = 0.7, r = 6: r drift {:.1e}, Omega/Omega_BPT - 1 = {:.1e}".format(
            np.ptp(dyn.r), Om / ham.circ_omega(6.0) - 1
        )
    )

    # 3. inclined orbit around a spinning black hole, no radiation reaction: E conserved
    dyn = Dynamics(
        0.0,
        ham=hamiltonian.Kerr([0.5, 0.0, 0.5]),
        ics="spherical",
        ic_pars={"r0": 8.0, "iota": 0.6},
        tmax=2000,
        dt=1.0,
    )
    dyn.evolve()
    assert np.ptp(dyn.E) < 1e-8 and np.ptp(dyn.Lz) < 1e-7 and np.ptp(dyn.Q) < 1e-6
    print(
        "Kerr, tilted spin, inclined orbit: E drift {:.1e}, r in [{:.3f}, {:.3f}]".format(
            np.ptp(dyn.E), dyn.r.min(), dyn.r.max()
        )
    )

    # 4. generic (eccentric, inclined) geodesics: conservation of E, L_z and the Carter constant Q
    for avec in ([0.0, 0.0, 0.9], [0.5, 0.3, 0.6], [0.0, 0.0, 0.0]):
        ham_g = hamiltonian.Kerr(avec)
        x0, p0 = ic.ICs(
            0.0,
            ham=ham_g,
            flux=flx.NoFlux(),
            kind="spherical",
            ic_pars={"r0": 10.0, "iota": 0.7},
        ).__ic__()
        p0 = 0.93 * p0 + np.array(
            [0.02, 0.0, 0.0]
        )  # less angular momentum, some radial momentum: eccentric
        dyn = Dynamics(
            0.0,
            ham=ham_g,
            ics="cartesian",
            ic_pars={"x0": x0, "p0": p0},
            tmax=20000,
            dt=1.0,
        )
        dyn.evolve()
        drift = [
            np.ptp(q) / max(abs(np.mean(q)), 1e-30) for q in (dyn.E, dyn.Lz, dyn.Q)
        ]
        theta = np.degrees(
            np.arccos(
                dyn.x
                @ (ham_g.avec / ham_g.a if ham_g.a else np.array([0, 0, 1.0]))
                / dyn.r
            )
        )
        print(
            "geodesic, a = {}: r in [{:.2f}, {:.2f}], theta in [{:.0f}, {:.0f}] deg, {:.0f} orbits; relative drift of E, Lz, Q: {:.1e}, {:.1e}, {:.1e}".format(
                avec,
                dyn.r.min(),
                dyn.r.max(),
                theta.min(),
                theta.max(),
                dyn.t[-1]
                * np.mean(np.linalg.norm(dyn.v, axis=1) / dyn.r)
                / (2 * np.pi),
                *drift
            )
        )
        assert max(drift) < 1e-6

    # 5. quasi-circular inspiral and plunge with radiation reaction
    nu = 0.05
    dyn = Dynamics(
        nu,
        ham=ham,
        flux=flx.QuadrupoleFlux(nu),
        ics="circular",
        ic_pars={"r0": 1.6 * ham.r_isco()},
        verbose=True,
    )
    dyn.evolve()
    print(
        "inspiral from 1.6 r_ISCO: {:.1f} orbits, plunge at t = {:.0f} M".format(
            (dyn.phi()[-1] - dyn.phi()[0]) / (2 * np.pi), dyn.t[-1]
        )
    )
    print("ISCO(0.7), LR(0.7):", ham.r_isco(), ham.r_lightring())
