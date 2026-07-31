"""make_data.py — regenerate the `const D = {...}` blob in index.html.

Ramp-flat geometry (fig1 alpha-partition style), replacing the earlier
smooth exponential+arc fault:

  * plate interface: planar ramp at THETA_R from the trench (0, 0) down to
    the decollement depth Z_D, then a horizontal flat out to X_END.  A
    node sits exactly at the bend, so the fold (axial-surface) dislocation
    network contains exactly ONE nonzero element: the hanging wall has one
    axial surface, rooted at the ramp-flat bend.
  * slab (footwall): flat incoming plate for x < 0, then straight down the
    ramp plane forever — the footwall bends once, at the trench.
  * long-term uplift v_struct: a BOX, sin(THETA_R) between the trench and
    the axial surface's surface intercept x_a, zero beyond.

Elastic Green's functions (fault_u2 per segment, fold_u2 per interior
node) follow the paper's construction (final/schematic/discrete_model.py
coseismic_discrete_coupled + the u2 normalization of fig1_alpha_partition,
surface step = slip * sin(dip)):

  * fault_u2[i]  = u2_segment(node_i -> node_i+1, signed_slip = -1)
  * fold_u2[j]   = u2_segment(axial intercept -> bend, signed_slip = -dtheta)
    and the fold dislocations are ALWAYS fully locked - they slip only
    coseismically, so the in-page JS adds them independent of the lock
    point, scaled by the hanging-wall share (1 - alpha):
        v_co = sum(locked fault segments) + (1 - alpha) * sum(fold_u2)
    which makes the interseismic field smooth (the concentrated fold
    steps cancel exactly against the long-term box).

Writes the blob in place (line `const D = ...;`) of index.html.
Run:  python3 make_data.py
"""

import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


# ── half-space edge dislocation, paper normalization (fig1/continuous) ──
def u2_edge(x_obs, slip, delta, d, x_ref, orient):
    if d <= 0:
        return -(slip / np.pi) * np.sin(delta) * np.arctan2(
            orient * (x_obs - x_ref), 1e-12)
    zeta = orient * (x_obs - x_ref) / d
    return -(slip / np.pi) * (
        np.sin(delta) * np.arctan(zeta)
        + (np.cos(delta) + zeta * np.sin(delta)) / (1.0 + zeta**2))


def u2_segment(x_obs, x1, z1, x2, z2, signed_slip):
    d1, d2 = -z1, -z2
    dx, dd = x2 - x1, d2 - d1
    delta = np.arctan2(abs(dd), max(abs(dx), 1e-12))
    orient = np.sign(dx) if abs(dx) > 0 else 1.0
    if d1 <= d2:
        xt, dt, xb, db = x1, d1, x2, d2
    else:
        xt, dt, xb, db = x2, d2, x1, d1
    return (u2_edge(x_obs, signed_slip, delta, dt, xt, orient)
            + u2_edge(x_obs, -signed_slip, delta, db, xb, orient))

# ── geometry constants ──────────────────────────────────────────────────
THETA_R = np.radians(20.0)   # ramp dip
Z_D     = -50.0              # decollement depth (km)
X_END   = 300.0              # section end (km)
X_MIN   = -50.0              # incoming-plate start (km)
X_K     = -Z_D / np.tan(THETA_R)          # ramp-flat bend x  (137.37 km)
GAMMA   = 0.5 * (-THETA_R + 0.0 + np.pi)  # axial bisector inclination (80 deg)
X_A     = X_K - Z_D / np.tan(GAMMA)       # axial surface intercept (146.19 km)

ALPHA0, LOCK0 = 0.5, 80.0
NSEG_RAMP, NSEG_FLAT = 190, 210           # node 190 lands exactly on the bend
NSEG = NSEG_RAMP + NSEG_FLAT
M_SURF = 750

r5 = lambda a: [round(float(v), 5) for v in np.ravel(a)]


def fault_z(x):
    """Interface elevation: 0 landward of trench edge cases handled by caller."""
    x = np.asarray(x, float)
    return np.where(x <= 0, 0.0, np.where(x <= X_K, -x * np.tan(THETA_R), Z_D))


def main():
    # ── fault polyline [x, z, theta_positive] ───────────────────────────
    xf = np.concatenate([np.linspace(X_MIN, 0.0, 50, endpoint=False),
                         np.linspace(0.0, X_END, 1000)])
    zf = fault_z(xf)
    tf = np.where((xf > 0) & (xf < X_K), THETA_R, 0.0)
    fault = [[round(float(x), 3), round(float(z), 3), round(float(t), 5)]
             for x, z, t in zip(xf, zf, tf)]

    # ── slab polyline: flat ocean plate, then straight down the ramp plane ──
    xs = np.concatenate([np.linspace(X_MIN, 0.0, 100, endpoint=False),
                         np.linspace(0.0, X_END, 900)])
    zs = np.where(xs <= 0, 0.0, -xs * np.tan(THETA_R))
    ts = np.where(xs <= 0, 0.0, THETA_R)
    slab = [[round(float(x), 3), round(float(z), 3), round(float(t), 5)]
            for x, z, t in zip(xs, zs, ts)]

    # ── surface observation grid ────────────────────────────────────────
    x_surf = np.linspace(0.5, X_END, M_SURF)

    # ── theta LUT over the hanging wall (Suppe domains, 1 axial surface) ──
    nx, nz = 241, 61
    gx = np.linspace(0.0, X_END, nx)
    gz = np.linspace(-60.0, 0.0, nz)
    theta_grid = np.zeros((nx, nz))
    for i, x in enumerate(gx):
        for j, z in enumerate(gz):
            if z < fault_z(x):            # below the interface: not hanging wall
                continue
            x_ax = X_K + (z - Z_D) / np.tan(GAMMA)   # axial surface at this depth
            theta_grid[i, j] = THETA_R if x < x_ax else 0.0
    theta_lut = dict(nx=nx, nz=nz, x0=0.0, x1=X_END, z0=-60.0, z1=0.0,
                     theta=r5(theta_grid))

    # ── fault nodes: bend node exact, uniform along each limb ───────────
    xn_ramp = np.linspace(0.0, X_K, NSEG_RAMP + 1)
    xn_flat = np.linspace(X_K, X_END, NSEG_FLAT + 1)[1:]
    xnode = np.concatenate([xn_ramp, xn_flat])
    znode = fault_z(xnode)
    theta_lib = np.arctan2(np.diff(znode), np.diff(xnode))   # negative on ramp
    x_mid = 0.5 * (xnode[:-1] + xnode[1:])

    # axial intercepts (bookkeeping only; the JS uses just x_mid + GFs)
    x_axial = []
    for jn in range(1, NSEG):
        g = 0.5 * (theta_lib[jn - 1] + theta_lib[jn] + np.pi)
        tg = np.tan(g)
        x_axial.append(xnode[jn] - znode[jn] / tg if abs(tg) < 1e12
                       else xnode[jn])

    # ── elastic GFs, paper convention (coseismic_discrete_coupled) ──────
    # fault: signed slip -1 per unit-slip segment
    fault_u2 = np.empty((NSEG, M_SURF))
    for i in range(NSEG):
        fault_u2[i] = u2_segment(x_surf, xnode[i], znode[i],
                                 xnode[i + 1], znode[i + 1], -1.0)

    # fold: zero except the single bend node; slip fs = s0 * dtheta along
    # the axial segment from its surface intercept to the bend, signed -fs
    fold_u2 = np.zeros((NSEG - 1, M_SURF))
    for jn in range(1, NSEG):
        dth = theta_lib[jn] - theta_lib[jn - 1]
        if abs(dth) < 1e-12:
            continue
        theta_avg = 0.5 * (theta_lib[jn - 1] + theta_lib[jn])
        t = -znode[jn] / np.cos(theta_avg)
        xa = xnode[jn] + t * (-np.sin(theta_avg))
        fold_u2[jn - 1] = u2_segment(x_surf, xa, 0.0,
                                     xnode[jn], znode[jn], -dth)

    # ── long-term kinematic uplift: the box ─────────────────────────────
    v_struct = np.where(x_surf < X_A, np.sin(THETA_R), 0.0)

    D = dict(
        alpha0=ALPHA0, lock0=LOCK0,
        fault=fault, slab=slab,
        x_surf=r5(x_surf),
        theta_lut=theta_lut,
        segments=dict(nseg=NSEG, m=M_SURF,
                      xnode=r5(xnode), znode=r5(znode),
                      theta=r5(np.abs(theta_lib)),
                      x_mid=r5(x_mid), x_axial=r5(np.array(x_axial)),
                      fault_u2=r5(fault_u2), fold_u2=r5(fold_u2),
                      v_struct=r5(v_struct)),
    )

    blob = "const D = " + json.dumps(D, separators=(",", ":")) + ";"
    html_path = os.path.join(HERE, "index.html")
    lines = open(html_path).readlines()
    hits = [k for k, ln in enumerate(lines)
            if re.match(r"\s*const D = ", ln)]
    assert len(hits) == 1, hits
    lines[hits[0]] = blob + "\n"
    open(html_path, "w").writelines(lines)
    print(f"geometry: ramp {np.degrees(THETA_R):.0f} deg, bend x={X_K:.2f}, "
          f"axial intercept x_a={X_A:.2f}, box height {np.sin(THETA_R):.3f}")
    print(f"wrote blob ({len(blob)/1e6:.2f} MB) into {html_path}")


if __name__ == "__main__":
    main()
