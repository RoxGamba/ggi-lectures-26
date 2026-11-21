"""
Animations for toy model 1 (a particle on Kerr with quadrupolar radiation reaction).

  toy_plunge       quasi-circular inspiral and plunge on Schwarzschild: orbit, effective potential, l = 2 modes
  toy_hangup       aligned spin a = -0.6, 0, +0.6, same initial frequency: the orbital hang-up
  toy_ecc          eccentric orbit on Kerr: the rosette, the particle bouncing in W(r; j), and the bursts
  toy_capture      same energy, three angular momenta: plunge, capture, scattering
  toy_prec         inclined orbit on Kerr: the orbital plane and L precess around the spin; mode content
  toy_prec_frames  same orbit: the waveform in the inertial, co-precessing and co-orbital frames

    python animations.py [plunge] [hangup] [ecc] [capture] [prec] [frames] [--outdir DIR]     (default: all)

Writes DIR/<name>.mp4 (default DIR: ../../movies/new). Needs ffmpeg: on the PATH, or FFMPEG=/path/to/ffmpeg,
or the imageio-ffmpeg package.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))  # style.py
from style import C, INK, MUTED  # noqa: E402

import hamiltonian  # noqa: E402
import flux as flx  # noqa: E402
import dynamics as dyn  # noqa: E402
import waveform  # noqa: E402

NU = 0.05
FPS, HOLD = 12, 2.0  # frames per second; seconds on the last frame
WIDTH = 1920  # pixels
DOT = dict(marker="o", ms=8, mec="white", mew=1.5, ls="none")
TH = np.linspace(0, 2 * np.pi, 200)


# helpers
def run(nu, a, ics, ic_pars, **kw):
    """Dynamics and l = 2 modes (mass and current quadrupole; mass quadrupole only in h22m)."""
    d = dyn.Dynamics(
        nu,
        ham=hamiltonian.Kerr(a),
        flux=flx.QuadrupoleFlux(nu),
        ics=ics,
        ic_pars=ic_pars,
        **kw,
    )
    d.evolve()
    return d


def modes(d, current=True):
    wf = waveform.QuadrupoleWaveform(d, current=current)
    wf.compute_modes([(2, m) for m in range(-2, 3)])
    return wf


def W(ham, r, j):
    """Effective potential for equatorial orbits: the energy of a particle with p_r = 0 and p_phi = j."""
    r = np.atleast_1d(np.asarray(r, float))
    z = np.zeros_like(r)
    return ham.energy(np.stack([r, z, z], 1), np.stack([z, j / r, z], 1))


def ffmpeg_exe():
    exe = os.environ.get("FFMPEG") or shutil.which("ffmpeg")
    if exe is None:
        try:
            import imageio_ffmpeg

            exe = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            raise RuntimeError(
                "ffmpeg not found: install it, set FFMPEG=/path/to/ffmpeg, or pip install imageio-ffmpeg"
            )
    return exe


def write(outdir, name, nframes, draw):
    """Render draw(n) for n < nframes and encode outdir/name.mp4 (H.264, the last frame held for HOLD seconds)."""
    outdir.mkdir(parents=True, exist_ok=True)
    out = outdir / f"{name}.mp4"
    with tempfile.TemporaryDirectory() as tmp:
        for n in range(nframes):
            fig = draw(n)
            with plt.rc_context(
                {"savefig.bbox": None}
            ):  # the whole figure: every frame has the same size
                fig.savefig(f"{tmp}/f{n:04d}.png", dpi=WIDTH / fig.get_figwidth())
            plt.close(fig)
        for k in range(int(HOLD * FPS)):
            shutil.copy(
                f"{tmp}/f{nframes - 1:04d}.png", f"{tmp}/f{nframes + k:04d}.png"
            )
        cmd = [
            ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-framerate",
            str(FPS),
            "-i",
            f"{tmp}/f%04d.png",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2:color=white",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            "-movflags",
            "+faststart",
            str(out),
        ]
        subprocess.run(cmd, check=True)
    print(f"wrote {out}  ({nframes} frames)")


def layout():
    fig = plt.figure(figsize=(9.0, 4.3))
    gs = fig.add_gridspec(
        2,
        2,
        width_ratios=[1, 1.25],
        left=0.07,
        right=0.985,
        bottom=0.13,
        top=0.94,
        wspace=0.30,
        hspace=0.42,
    )
    return (
        fig,
        fig.add_subplot(gs[:, 0]),
        fig.add_subplot(gs[0, 1]),
        fig.add_subplot(gs[1, 1]),
    )


# plunge
def plunge(outdir):
    """Frames equally spaced in orbital phase; the waveforms stop at the peak of |h22| (no ringdown attached)."""
    R0, N = 9.0, 110
    d = run(NU, 0.0, "circular", {"r0": R0}, dt=0.25, r_stop=1.02 * 2)
    ham, t, r, x, y, j = d.ham, d.t, d.r, d.x[:, 0], d.x[:, 1], d.L[:, 2]
    wf = modes(d)
    h = wf.hlm[(2, 2)]
    im = np.argmax(np.abs(h))
    tt = t[: im + 1]
    hh, hh21, hh20 = (wf.hlm[(2, m)][: im + 1].real for m in (2, 1, 0))
    print(
        "|h21|/|h22| at the peak: %.2f, |h20|/|h22|: %.2f"
        % (abs(wf.hlm[(2, 1)][im]) / abs(h[im]), abs(wf.hlm[(2, 0)][im]) / abs(h[im]))
    )
    phi = d.phi()
    print("orbits: %.1f, t_peak = %.0f M" % (phi[-1] / (2 * np.pi), t[im]))
    frames = [
        min(int(np.searchsorted(phi, f)), t.size - 1)
        for f in np.linspace(phi[0], phi[-1], N)
    ]
    rr = np.linspace(2.02, 14, 600)
    r_isco, r_lr = ham.r_isco(), ham.r_lightring()

    def draw(n):
        k = frames[n]
        fig = plt.figure(figsize=(9.0, 4.3))
        gs = fig.add_gridspec(
            3,
            2,
            width_ratios=[1, 1.25],
            height_ratios=[1.15, 1, 1],
            left=0.06,
            right=0.985,
            bottom=0.13,
            top=0.95,
            wspace=0.24,
            hspace=0.18,
        )
        ao, aw, ah = (
            fig.add_subplot(gs[:, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[1, 1]),
        )
        ah2 = fig.add_subplot(gs[2, 1], sharex=ah)

        ao.fill(ham.rH * np.cos(TH), ham.rH * np.sin(TH), color=INK)
        for rad, ls in ((r_lr, (0, (2, 2))), (r_isco, (0, (5, 3)))):
            ao.plot(rad * np.cos(TH), rad * np.sin(TH), color=MUTED, lw=0.9, ls=ls)
        ao.text(0, r_isco + 0.35, "ISCO", color=INK, fontsize=10, ha="center")
        ao.text(0, r_lr + 0.3, "light ring", color=INK, fontsize=10, ha="center")
        ao.plot(x[: k + 1 : 4], y[: k + 1 : 4], color=C[0], lw=0.9, alpha=0.55)
        ao.plot(x[k], y[k], color=C[1], **DOT)
        ao.set(
            xlim=(-10, 10),
            ylim=(-10, 10),
            aspect="equal",
            xlabel="$x/M$",
            ylabel="$y/M$",
        )
        ao.grid(False)

        aw.plot(rr, W(ham, rr, j[0]), color=MUTED, lw=1.0, ls=(0, (4, 2)))
        aw.plot(rr, W(ham, rr, j[k]), color=C[0])
        aw.plot(r[k], d.E[k], color=C[1], **DOT)
        aw.set(xlim=(2, 14), ylim=(0.905, 0.975), ylabel=r"$W(r;p_\varphi)$")
        aw.xaxis.set_label_position("top")
        aw.xaxis.tick_top()
        aw.set_xlabel("$r/M$")
        aw.text(
            0.97,
            0.9,
            rf"$p_\varphi={j[k]:.2f}$",
            transform=aw.transAxes,
            ha="right",
            va="top",
            color=INK,
        )

        m = tt <= t[k]
        ah.plot(tt[m] - t[im], hh[m], color=C[0], lw=1.2)
        ah.set(xlim=(tt[0] - t[im], 40), ylim=(-1.9, 1.9), ylabel=r"$\Re h_{22}$")
        ah.tick_params(labelbottom=False)
        ah2.plot(tt[m] - t[im], hh21[m], color=C[1], lw=1.2)
        ah2.plot(tt[m] - t[im], hh20[m], color=C[2], lw=1.2)
        ah2.set(
            ylim=(-0.34, 0.34), xlabel=r"$(t-t_{\rm peak})/M$", ylabel=r"$\Re h_{2m}$"
        )
        ah2.text(
            0.02,
            0.86,
            "$(2,1)$",
            transform=ah2.transAxes,
            color=C[1],
            fontsize=11,
            va="top",
        )
        ah2.text(
            0.14,
            0.86,
            "$(2,0)$",
            transform=ah2.transAxes,
            color=C[2],
            fontsize=11,
            va="top",
        )
        return fig

    write(outdir, "toy_plunge", N, draw)


# hang-up
def hangup(outdir):
    """Three runs with the same initial orbital frequency, a = -0.6, 0, +0.6, synchronised in time."""
    OM0, N = 0.03, 150
    runs = []
    for a, c in ((-0.6, C[1]), (0.0, MUTED), (0.6, C[0])):
        d = run(NU, a, "circular", {"r0": (1 / OM0 - a) ** (2 / 3)}, dt=1.0)
        h = modes(d, current=False).hlm[(2, 2)].real
        ph = d.phi()
        runs.append(dict(a=a, c=c, d=d, h=h, n=(ph[-1] - ph[0]) / (2 * np.pi)))
        print(f"a = {a:+.1f}: {runs[-1]['n']:.1f} orbits, t_end = {d.t[-1]:.0f} M")
    t_end = max(R["d"].t[-1] for R in runs)
    times = np.append(
        np.linspace(0, t_end, N - 1), t_end + 1.0
    )  # last frame: all runs finished
    Emin = min(R["d"].E[-1] for R in runs)
    hmax = 1.15 * max(np.abs(R["h"]).max() for R in runs)
    rr = np.linspace(1.2, 16, 800)

    def draw(n):
        tnow = times[n]
        fig = plt.figure(figsize=(9.0, 5.0))
        gs = fig.add_gridspec(
            2,
            3,
            height_ratios=[1.15, 1],
            left=0.07,
            right=0.985,
            bottom=0.1,
            top=0.94,
            wspace=0.28,
            hspace=0.32,
        )
        gb = gs[1, :].subgridspec(1, 2, width_ratios=[1, 1.5], wspace=0.25)
        aw = fig.add_subplot(gb[0])
        ah = [fig.add_subplot(g) for g in gb[1].subgridspec(3, 1, hspace=0.08)]
        for i, R in enumerate(runs):
            d, a, c = R["d"], R["a"], R["c"]
            ham = d.ham
            k = min(int(np.searchsorted(d.t, tnow)), d.t.size - 1)
            alive = tnow <= d.t[-1]
            cl = INK if c == MUTED else c

            ao = fig.add_subplot(gs[0, i])
            ao.fill(ham.rH * np.cos(TH), ham.rH * np.sin(TH), color=INK)
            ri = ham.r_isco()
            ao.plot(
                ri * np.cos(TH), ri * np.sin(TH), color=MUTED, lw=0.9, ls=(0, (5, 3))
            )
            ao.plot(d.x[: k + 1, 0], d.x[: k + 1, 1], color=cl, lw=0.7, alpha=0.5)
            if alive:
                ao.plot(d.x[k, 0], d.x[k, 1], "o", color=cl, ms=7, mec="white", mew=1.2)
            ao.set(
                xlim=(-12, 12),
                ylim=(-12, 12),
                aspect="equal",
                xticks=[-10, 0, 10],
                yticks=[-10, 0, 10],
            )
            ao.set_title(
                rf"$a={a:+.1f}$:  $r_{{\rm ISCO}}={ri:.1f}M$",
                fontsize=11,
                color=INK,
                pad=3,
            )
            ao.grid(False)
            ao.tick_params(labelsize=8)

            m = rr > 1.05 * ham.rH
            aw.plot(rr[m], W(ham, rr[m], d.L[k, 2]), color=cl, lw=1.4)
            if alive:
                aw.plot(d.r[k], d.E[k], "o", color=cl, ms=7, mec="white", mew=1.2)

            mk = d.t <= tnow
            ah[i].plot(d.t[mk], R["h"][mk], color=cl, lw=0.9)
            ah[i].set(xlim=(0, t_end), ylim=(-hmax, hmax), yticks=[])
            ah[i].text(
                0.005,
                0.92,
                rf"$a={a:+.1f}$",
                transform=ah[i].transAxes,
                fontsize=9,
                color=cl,
                va="top",
            )
            if not alive:
                ah[i].text(
                    0.995,
                    0.92,
                    f"{R['n']:.0f} orbits",
                    transform=ah[i].transAxes,
                    fontsize=9,
                    color=cl,
                    va="top",
                    ha="right",
                )
            if i < 2:
                ah[i].tick_params(labelbottom=False)
        aw.set(
            xlim=(1.2, 16),
            ylim=(Emin - 0.01, 0.985),
            xlabel="$r/M$",
            ylabel=r"$W(r;p_\varphi)$",
        )
        aw.tick_params(labelsize=8)
        ah[1].set_ylabel(r"$\Re h_{22}$")
        ah[2].set_xlabel("$t/M$")
        ah[2].tick_params(labelsize=8)
        return fig

    write(outdir, "toy_hangup", N, draw)


# eccentric
def ecc(outdir):
    a, N = 0.5, 150
    d = run(NU, a, "eccentric", {"p": 14.0, "e": 0.6}, dt=0.5)
    ham = d.ham
    h = modes(d, current=False).hlm[(2, 2)].real
    t, r, j = d.t, d.r, d.L[:, 2]
    tmax = min(3000.0, t[-1])
    idx = [int(np.searchsorted(t, s)) for s in np.linspace(0, tmax, N)]
    rr = np.linspace(ham.rH * 1.02, 45, 800)
    lim = 1.08 * r[t <= tmax].max()
    hlim = 1.05 * np.abs(h[t <= tmax]).max()

    def draw(n):
        k = idx[n]
        fig, ao, aw, ah = layout()
        lo = max(0, k - 1400)
        ao.plot(
            d.x[: k + 1 : 3, 0], d.x[: k + 1 : 3, 1], color=C[0], lw=0.8, alpha=0.35
        )
        ao.plot(d.x[lo : k + 1, 0], d.x[lo : k + 1, 1], color=C[0], lw=1.6)
        ao.plot(0, 0, "o", color=INK, ms=7)
        ao.plot(d.x[k, 0], d.x[k, 1], color=C[1], **DOT)
        ao.set(
            xlim=(-lim, lim),
            ylim=(-lim, lim),
            aspect="equal",
            xlabel="$x/M$",
            ylabel="$y/M$",
        )
        ao.grid(False)
        aw.plot(rr, W(ham, rr, j[k]), color=C[0])
        aw.axhline(d.E[k], color=MUTED, lw=0.9, ls=(0, (4, 2)))
        aw.plot(r[k], d.E[k], color=C[1], **DOT)
        aw.set(xlim=(2, 42), ylim=(0.94, 1.0), xlabel="$r/M$", ylabel=r"$W$, $E$")
        m = t <= t[k]
        ah.plot(t[m], h[m], color=C[0], lw=1.2)
        ah.set(
            xlim=(0, tmax), ylim=(-hlim, hlim), xlabel="$t/M$", ylabel=r"$\Re h_{22}$"
        )
        return fig

    write(outdir, "toy_ecc", N, draw)


# capture
def capture(outdir):
    nu, E0, r0, N = 0.1, 1.004, 80.0, 150
    runs = []
    for j0, lab, c in (
        (4.0, "plunge", C[1]),
        (4.1, "capture", C[0]),
        (4.5, "scattering", C[2]),
    ):
        d = run(
            nu,
            0.0,
            "energy",
            {"E0": E0, "j0": j0, "r0": r0},
            tmax=3e4,
            dt=0.5,
            r_max=120.0,
        )
        runs.append((j0, lab, c, d, np.abs(modes(d, current=False).hlm[(2, 2)])))
    tmax = 3100.0
    times = np.linspace(0, tmax, N)
    rr = np.geomspace(2.05, 90, 600)

    def draw(n):
        tn = times[n]
        fig, ao, aw, ah = layout()
        for j0, lab, c, d, A in runs:
            k = min(int(np.searchsorted(d.t, tn)), d.t.size - 1)
            ao.plot(d.x[: k + 1, 0], d.x[: k + 1, 1], color=c, lw=1.4, label=lab)
            aw.plot(rr, W(d.ham, rr, j0), color=c, lw=1.0, ls=(0, (4, 2)))
            aw.plot(d.r[: k + 1], d.E[: k + 1], color=c, lw=2.0)
            if tn <= d.t[-1]:
                ao.plot(d.x[k, 0], d.x[k, 1], color=c, **DOT)
                aw.plot(d.r[k], d.E[k], color=c, **DOT)
            ah.plot(d.t[: k + 1], A[: k + 1], color=c, lw=1.4)
        ao.plot(0, 0, "o", color=INK, ms=7)
        ao.set(
            xlim=(-45, 85),
            ylim=(-55, 55),
            aspect="equal",
            xlabel="$x/M$",
            ylabel="$y/M$",
        )
        ao.legend(loc="lower right")
        ao.grid(False)
        aw.axhline(1, color=MUTED, lw=0.8)
        aw.set(
            xscale="log",
            xlim=(2, 90),
            ylim=(0.93, 1.03),
            xlabel="$r/M$",
            ylabel=r"$E$, $W$",
        )
        aw.set_xticks([2, 5, 10, 20, 50])
        aw.set_xticklabels(["2", "5", "10", "20", "50"])
        ah.set(xlim=(0, tmax), ylim=(0, 1.7), xlabel="$t/M$", ylabel=r"$|h_{22}|$")
        return fig

    write(outdir, "toy_capture", N, draw)


# precession
def precessing_run():
    return run(NU, 0.8, "spherical", {"r0": 10.0, "iota": np.radians(45)}, dt=1.0)


def orbit3d(ao, d, k):
    lo = max(0, k - 160)
    ao.plot(*d.x[: k + 1].T, color=C[0], lw=0.6, alpha=0.3)
    ao.plot(*d.x[lo : k + 1].T, color=C[0], lw=1.6)
    ao.plot(*[[v] for v in d.x[k]], color=C[1], **DOT)
    ao.quiver(0, 0, 0, 0, 0, 9, color=INK, lw=1.8, arrow_length_ratio=0.12)
    ao.text(0, 0, 10.3, r"$\mathbf{a}$", color=INK, fontsize=13)
    ao.set(
        xlim=(-10, 10),
        ylim=(-10, 10),
        zlim=(-8, 10),
        xlabel="$x/M$",
        ylabel="$y/M$",
        zlabel="$z/M$",
    )
    ao.set_box_aspect((1, 1, 0.9))
    ao.view_init(elev=22, azim=-60)


def prec(outdir):
    N = 130
    d = precessing_run()
    hm = modes(d).hlm
    rms = lambda z: np.sqrt(uniform_filter1d(np.abs(z) ** 2, 200))
    t = d.t
    R = {m: rms(hm[(2, m)]) for m in (2, 1, 0)}
    tmax = min(1300.0, t[-1] - 200)
    idx = [int(np.searchsorted(t, s)) for s in np.linspace(150, tmax, N)]
    Lhat = (
        d.L / np.linalg.norm(d.L, axis=1)[:, None]
    )  # canonical L = x cross p: smooth precession
    LN = np.cross(d.x, d.v)  # normal to the instantaneous orbital plane
    LNhat = (
        LN / np.linalg.norm(LN, axis=1)[:, None]
    )  # nutates twice per orbit (frame dragging)
    hmax = 1.15 * np.abs(hm[(2, 2)][(t > 150) & (t < tmax)]).max()

    def draw(n):
        k = idx[n]
        fig = plt.figure(figsize=(9.0, 4.3))
        gs = fig.add_gridspec(
            2,
            2,
            width_ratios=[1, 1.15],
            left=0.0,
            right=0.985,
            bottom=0.13,
            top=0.94,
            wspace=0.16,
            hspace=0.42,
        )
        ao = fig.add_subplot(gs[:, 0], projection="3d")
        am, aa = fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 1])
        orbit3d(ao, d, k)
        ao.quiver(0, 0, 0, *(9 * LNhat[k]), color=C[2], lw=1.4, arrow_length_ratio=0.12)
        ao.plot(*(9 * LNhat[: k + 1]).T, color=C[2], lw=0.8, alpha=0.8)
        ao.quiver(0, 0, 0, *(9 * Lhat[k]), color=C[1], lw=1.8, arrow_length_ratio=0.12)
        ao.plot(*(9 * Lhat[: k + 1]).T, color=C[1], lw=0.8, alpha=0.6)
        ao.text2D(
            0.03,
            0.95,
            r"$\mathbf{L}=\mathbf{x}\times\mathbf{p}$",
            transform=ao.transAxes,
            color=C[1],
            fontsize=12,
        )
        ao.text2D(
            0.03,
            0.87,
            r"$\hat{\mathbf{L}}_N\propto\mathbf{x}\times\mathbf{v}$",
            transform=ao.transAxes,
            color=C[2],
            fontsize=12,
        )
        m = (t >= 150) & (t <= t[k])
        for mm, c in ((2, C[0]), (1, C[1]), (0, C[2])):
            am.plot(t[m], R[mm][m], color=c, label=rf"$m={mm}$")
        am.set(
            xlim=(150, tmax),
            ylim=(0, 0.8),
            xlabel="$t/M$",
            ylabel=r"$\langle|h_{2m}|^2\rangle^{1/2}$",
        )
        am.legend(loc="upper left", ncol=3)
        aa.plot(t[m], hm[(2, 2)].real[m], color=C[0], lw=1.0)
        aa.axhline(0, color=MUTED, lw=0.8)
        aa.set(
            xlim=(150, tmax),
            ylim=(-hmax, hmax),
            xlabel="$t/M$",
            ylabel=r"$\Re\,h_{22}$ (inertial frame)",
        )
        return fig

    write(outdir, "toy_prec", N, draw)


def frames(outdir):
    N = 150
    d = precessing_run()
    t = d.t
    wf = modes(d)
    fr = wf.coprecessing()
    h_in, h_cp, h_co, R = wf.hlm, fr["cp"], fr["co"][(2, 2)], fr["R"]
    t0, tmax = 150.0, t[-1] - 150
    idx = [int(np.searchsorted(t, s_)) for s_ in np.linspace(t0 + 20, tmax, N)]

    def modes_panel(ax, h, m, label):
        ax.plot(t[m], h[(2, 2)].real[m], color=MUTED, lw=0.5)
        for mm, c in ((2, C[0]), (1, C[1]), (0, C[2])):
            ax.plot(
                t[m], np.abs(h[(2, mm)])[m], color=c, lw=1.4, label=rf"$|h_{{2{mm}}}|$"
            )
        ax.text(
            0.01,
            0.95,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color=INK,
        )
        ax.set(xlim=(t0, tmax), ylim=(-1.25, 1.25))

    def draw(n):
        k = idx[n]
        fig = plt.figure(figsize=(9.0, 4.3))
        gs = fig.add_gridspec(
            3,
            2,
            width_ratios=[1, 1.2],
            left=0.0,
            right=0.985,
            bottom=0.11,
            top=0.97,
            wspace=0.24,
            hspace=0.12,
        )
        ao = fig.add_subplot(gs[:, 0], projection="3d")
        a1 = fig.add_subplot(gs[0, 1])
        a2, a3 = fig.add_subplot(gs[1, 1], sharex=a1), fig.add_subplot(
            gs[2, 1], sharex=a1
        )
        orbit3d(ao, d, k)
        # co-precessing triad: z' = L_hat (orange), x', y' (grey)
        for col, c, lab in (
            (2, C[1], r"$\hat{\mathbf{L}}_N$"),
            (0, MUTED, "$x'$"),
            (1, MUTED, "$y'$"),
        ):
            v = (9 if col == 2 else 6) * R[k][:, col]
            ao.quiver(
                0,
                0,
                0,
                *v,
                color=c,
                lw=1.8 if col == 2 else 1.2,
                arrow_length_ratio=0.12,
            )
            ao.text(*(1.12 * v), lab, color=c, fontsize=12)
        m = (t >= t0) & (t <= t[k])
        modes_panel(a1, h_in, m, "inertial frame")
        modes_panel(a2, h_cp, m, r"co-precessing frame (follows $\hat{\mathbf{L}}_N$)")
        a1.legend(loc="lower right", ncol=3, fontsize=9)
        a3.plot(t[m], h_co.real[m], color=C[0], lw=1.4, label=r"$\Re\,h_{22}$")
        a3.plot(t[m], h_co.imag[m], color=C[3], lw=1.4, label=r"$\Im\,h_{22}$")
        a3.text(
            0.01,
            0.95,
            "co-orbital frame (orbital phase removed too)",
            transform=a3.transAxes,
            ha="left",
            va="top",
            fontsize=10,
            color=INK,
        )
        a3.set(xlim=(t0, tmax), ylim=(-1.4, 1.0), xlabel="$t/M$")
        a3.legend(loc="center right", ncol=2, fontsize=9)
        for ax in (a1, a2):
            plt.setp(ax.get_xticklabels(), visible=False)
        return fig

    write(outdir, "toy_prec_frames", N, draw)


ALL = dict(
    plunge=plunge, hangup=hangup, ecc=ecc, capture=capture, prec=prec, frames=frames
)

if __name__ == "__main__":
    args = sys.argv[1:]
    outdir = HERE.parents[1] / "movies" / "new"
    if "--outdir" in args:
        i = args.index("--outdir")
        outdir = Path(args[i + 1]).resolve()
        args = args[:i] + args[i + 2 :]
    unknown = [a for a in args if a not in ALL]
    if unknown:
        sys.exit("unknown animation(s): {}. Choose among {}".format(unknown, list(ALL)))
    for name in args or list(ALL):
        ALL[name](outdir)
