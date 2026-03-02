#!/usr/bin/env python3
"""Axial surface convergence: circle vs sub-critical ellipse geometries.

Demonstrates how the structural uplift changes from a step function
(circle, κ/κ_crit = 1, all axial surfaces converge) to a smooth curve
(ellipse, κ/κ_crit < 1, axial surfaces distributed along surface).
"""

import sys
sys.path.insert(0, '/Users/braydennoh/Research/ThrustFault/2.28/quadrature')

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from continuous import SmoothFault, structural_velocity

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

# ═══════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════

depths = [10, 20, 30]           # km — lock depth = D
s0 = 20.0                       # mm/yr slip rate
x_plot = 100.0                   # km observation range
x_obs = np.linspace(2.0, x_plot, 500)

# Three columns: a/D ratio controls curvature relative to critical
# a/D = 1 → circle (κ/κ_crit = 1, axial surfaces converge)
# a/D > 1 → sub-critical (κ/κ_crit < 1, axial surfaces spread out)
cases = [
    (1.0, r'$a/D = 1$ (circle)'),
    (3.0, r'$a/D = 3$'),
    (5.0, r'$a/D = 5$'),
]
colors = ['steelblue', 'firebrick', '#228833']

# ═══════════════════════════════════════════════════════════════
# Figure: 3 columns × 2 rows (geometry + structural uplift)
# ═══════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(9, 5))
fig.subplots_adjust(hspace=0.05, wspace=0.25)

for col, (ad_ratio, title) in enumerate(cases):
    ax_geo = axes[0, col]
    ax_vel = axes[1, col]

    for D, clr in zip(depths, colors):
        a = ad_ratio * D
        f = SmoothFault.ellipse_flat(a, D, x_end=300.0)

        # Geometry
        xi = np.linspace(0, f.arc_length, 3000)
        xp, zp = f.evaluate(xi)
        ax_geo.plot(xp, zp, color=clr, lw=1.5, label=f'D = {D} km')

        # Structural uplift (kinematic axial-surface mapping)
        _, v = structural_velocity(x_obs, f, s0)
        ax_vel.plot(x_obs, v, color=clr, lw=1.2, label=f'D = {D} km')

    # Format geometry — box_aspect=0.7 gives equal data scaling (70/100)
    ax_geo.axhline(0, color='k', lw=0.8)
    ax_geo.set_xlim(0, x_plot)
    ax_geo.set_ylim(-60, 10)
    ax_geo.set_box_aspect(0.7)
    ax_geo.set_title(title)
    if col == 0:
        ax_geo.set_ylabel('Depth (km)')
        ax_geo.legend(fontsize=7, frameon=False)

    # Format velocity — same box dimensions as geometry row
    ax_vel.axhline(0, color='grey', lw=0.5, ls='--')
    ax_vel.set_xlim(0, x_plot)
    ax_vel.set_ylim(-2, 22)
    ax_vel.set_box_aspect(0.7)
    ax_vel.set_xlabel('x (km)')
    if col == 0:
        ax_vel.set_ylabel(r'Structural $\dot{v}$ (mm/yr)')
        ax_vel.legend(fontsize=7, frameon=False)

outpath = "/Users/braydennoh/Research/ThrustFault/2.28/quadrature/examples/axial_convergence.pdf"
plt.savefig(outpath, bbox_inches="tight")
plt.show()
print(f"Saved {outpath}")
