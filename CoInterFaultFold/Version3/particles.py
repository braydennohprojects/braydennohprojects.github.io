#!/usr/bin/env python3
"""Fault-bend fold: particle advection through sub/critical/super-critical bends.

At each step the velocity is set by the nearest axial surface to the RIGHT
(the most recently crossed), introducing them one at a time as the particle
moves leftward.  For super-critical bends, overlapping axial surfaces cause
non-physical velocity jumps and layer thickness changes.
"""

import sys
sys.path.insert(0, '/Users/braydennoh/Research/ThrustFault/2.28/quadrature')

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from continuous import SmoothFault

mpl.rcParams["figure.dpi"] = 150
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.titlesize": 10,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8,
})

D = 20.0
x_view = 50.0
x_start = 50.0
CALTECH_ORANGE = '#FF6C0C'

cases = [
    (2 * D, r'Sub-critical ($a/D = 2$)'),
    (D,     r'Critical ($a/D = 1$)'),
    (0.5*D, r'Super-critical ($a/D = 0.5$)'),
]


def particle_path(z0, f, xi_lock):
    """Integrate a particle from (x_start, z0) leftward.

    Velocity at each step is set by the nearest axial surface to the RIGHT
    of the particle (the most recently crossed).  Axial surfaces are
    recomputed at the particle's CURRENT depth each step.
    """
    if z0 >= -0.01:
        return np.array([x_start, -5]), np.array([0.0, 0.0])

    xi_dense = np.linspace(xi_lock + 40, 0.001, 800)
    xf_all, zf_all = f.evaluate(xi_dense)
    th_all = f.theta(xi_dense)

    x_curr, z_curr = x_start, z0
    path_x = [x_curr]
    path_z = [z_curr]
    dx = -0.15

    for _ in range(8000):
        if x_curr < -5 or z_curr >= 0.0:
            break

        keep = zf_all <= z_curr + 0.01
        if np.sum(keep) < 2:
            x_curr += dx
            path_x.append(x_curr)
            path_z.append(z_curr)
            continue

        xf = xf_all[keep]
        zf = zf_all[keep]
        th = th_all[keep]
        tan_th = np.clip(np.tan(th), -50, 50)
        x_ax = xf + (zf - z_curr) * tan_th

        # Nearest axial surface to the RIGHT = most recently crossed
        to_right = x_ax >= x_curr
        if np.any(to_right):
            idx_r = np.where(to_right)[0]
            i_active = idx_r[np.argmin(x_ax[to_right])]
            theta = float(th[i_active])
        else:
            # Left of all axial surfaces: use leftmost surface's θ
            i_leftmost = np.argmin(x_ax)
            theta = float(th[i_leftmost])

        dz = dx * np.clip(np.tan(theta), -50, 50)
        x_new = x_curr + dx
        z_new = z_curr + dz

        if z_new >= 0.0 and abs(dz) > 1e-15:
            frac = -z_curr / dz
            x_new = x_curr + frac * dx
            z_new = 0.0
            path_x.append(x_new)
            path_z.append(z_new)
            break

        x_curr = x_new
        z_curr = z_new
        path_x.append(x_curr)
        path_z.append(z_curr)

    return np.array(path_x), np.array(path_z)


fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
fig.subplots_adjust(wspace=0.22)

for col, (a, title) in enumerate(cases):
    ax = axes[col]
    f = SmoothFault.ellipse_flat(a, D, x_end=200.0)

    xi_d = np.linspace(0, f.arc_length, 5000)
    xd, _ = f.evaluate(xi_d)
    xi_lock = float(np.interp(a, xd, xi_d))

    # ── Axial surfaces (full range: curved + flat, ξ 0→500) ─
    n_ax = 1000
    xi_max = min(500, f.arc_length - 0.1)
    xi_ax = np.linspace(0.001, xi_max, n_ax)
    for xi_i in xi_ax:
        xfi = float(f.evaluate(xi_i)[0])
        zfi = float(f.evaluate(xi_i)[1])
        th_i = float(f.theta(xi_i))
        ct = np.cos(th_i)
        if abs(ct) < 1e-10:
            continue
        t = -zfi / ct
        x_s = xfi + t * (-np.sin(th_i))
        if -5 <= x_s <= x_view + 5:
            ax.plot([xfi, x_s], [zfi, 0], color='grey',
                    lw=0.3, alpha=0.4, zorder=2)

    # ── Particle paths ──────────────────────────────────────
    z_particles = np.linspace(-0.5, -D + 0.5, 20)
    paths = []
    for z0 in z_particles:
        xp, zp = particle_path(z0, f, xi_lock)
        paths.append((xp, zp))

    # ── Interpolate onto common x grid for fills ────────────
    x_grid = np.linspace(-5, x_start, 600)
    z_on_grid = []
    for xp, zp in paths:
        xp_inc = xp[::-1]
        zp_inc = zp[::-1]
        z_interp = np.interp(x_grid, xp_inc, zp_inc,
                             left=0.0, right=zp_inc[-1])
        z_interp = np.minimum(z_interp, 0.0)
        z_on_grid.append(z_interp)

    # ── Alternating Caltech orange / white fills ────────────
    for i in range(len(z_on_grid) - 1):
        color = CALTECH_ORANGE if i % 2 == 0 else 'white'
        ax.fill_between(x_grid, z_on_grid[i], z_on_grid[i + 1],
                        color=color, alpha=1.0, zorder=1, edgecolor='none')

    # ── Draw path boundary lines on top ─────────────────────
    for xp, zp in paths:
        ax.plot(xp, zp, 'k', lw=0.5, zorder=3)

    ax.axhline(0, color='k', lw=0.5, zorder=5)
    ax.set_xlim(-2, x_view)
    ax.set_ylim(-25, 0)
    ax.set_box_aspect(0.55)
    ax.set_title(title)
    ax.set_xlabel('x (km)')
    if col == 0:
        ax.set_ylabel('Depth (km)')

outpath = "/Users/braydennoh/Research/ThrustFault/2.28/quadrature/examples/particles.pdf"
plt.savefig(outpath, bbox_inches="tight")
plt.show()
print(f"Saved {outpath}")
