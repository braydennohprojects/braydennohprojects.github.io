#!/usr/bin/env python3
"""Circle + flat listric fault: curvature threshold demonstration.

The fault is a quarter circle (radius R, center at (R, 0)) from the
surface (0, 0) to the bottom (R, -R), followed by a horizontal flat
at z = -R.  On the arc, kappa = 1/R = kappa_crit exactly.

Usage:
    python continuous.py
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams["figure.dpi"] = 150
mpl.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 10,
})

# ═══════════════════════════════════════════════════════════════════
# Edge dislocation kernels
# ═══════════════════════════════════════════════════════════════════

def u1_edge(x_obs, slip, delta, d, x_ref, orient):
    if d <= 0:
        return -(slip / np.pi) * np.cos(delta) * np.arctan2(
            orient * (x_obs - x_ref), 1e-12)
    zeta = orient * (x_obs - x_ref) / d
    return -(slip / np.pi) * (
        np.cos(delta) * np.arctan(zeta)
        - (np.sin(delta) - zeta * np.cos(delta)) / (1.0 + zeta**2))


def u2_edge(x_obs, slip, delta, d, x_ref, orient):
    if d <= 0:
        return -(slip / np.pi) * np.sin(delta) * np.arctan2(
            orient * (x_obs - x_ref), 1e-12)
    zeta = orient * (x_obs - x_ref) / d
    return -(slip / np.pi) * (
        np.sin(delta) * np.arctan(zeta)
        + (np.cos(delta) + zeta * np.sin(delta)) / (1.0 + zeta**2))


def u1_segment(x_obs, x1, z1, x2, z2, signed_slip):
    d1, d2 = -z1, -z2
    dx, dd = x2 - x1, d2 - d1
    delta = np.arctan2(abs(dd), max(abs(dx), 1e-12))
    orient = np.sign(dx) if abs(dx) > 0 else 1.0
    if d1 <= d2:
        xt, dt, xb, db = x1, d1, x2, d2
    else:
        xt, dt, xb, db = x2, d2, x1, d1
    return (u1_edge(x_obs, signed_slip, delta, dt, xt, orient)
            + u1_edge(x_obs, -signed_slip, delta, db, xb, orient))


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


# ═══════════════════════════════════════════════════════════════════
# SmoothFault
# ═══════════════════════════════════════════════════════════════════

class SmoothFault:
    """Fault with analytically or numerically defined (x, z, theta, kappa)."""

    def __init__(self, x, z, theta=None, kappa=None):
        self._x = np.asarray(x, float)
        self._z = np.asarray(z, float)
        ds = np.sqrt(np.diff(self._x)**2 + np.diff(self._z)**2)
        self._s = np.concatenate(([0.0], np.cumsum(ds)))
        self.L = self._s[-1]

        if theta is not None:
            self._theta = np.asarray(theta, float)
        else:
            dx_ds = np.gradient(self._x, self._s)
            dz_ds = np.gradient(self._z, self._s)
            self._theta = np.arctan2(dz_ds, dx_ds)

        if kappa is not None:
            self._kappa = np.asarray(kappa, float)
        else:
            self._kappa = np.gradient(self._theta, self._s)

    @staticmethod
    def circle_flat(R, x_end=200.0, n_arc=3000, n_flat=3000):
        """Quarter circle (radius R, center (R,0)) + flat at z = -R."""
        # Arc: alpha from pi to 3pi/2 (counterclockwise, going down-right)
        alpha = np.linspace(np.pi, 1.5 * np.pi, n_arc)
        x_arc = R + R * np.cos(alpha)
        z_arc = R * np.sin(alpha)
        # theta = alpha - 3pi/2 maps to [-pi/2, 0]
        th_arc = alpha - 1.5 * np.pi
        k_arc = np.full(n_arc, 1.0 / R)

        # Flat: from (R, -R) to (x_end, -R)
        x_flat = np.linspace(R, x_end, n_flat + 1)[1:]   # skip duplicate
        z_flat = np.full(n_flat, -R)
        th_flat = np.zeros(n_flat)
        k_flat = np.zeros(n_flat)

        x = np.concatenate([x_arc, x_flat])
        z = np.concatenate([z_arc, z_flat])
        th = np.concatenate([th_arc, th_flat])
        k = np.concatenate([k_arc, k_flat])
        return SmoothFault(x, z, th, k)

    @staticmethod
    def ellipse_flat(a, D, x_end=300.0, n_arc=3000, n_flat=3000):
        """Elliptical arc (semi-axes a horiz, D vert, center (a,0)) + flat.

        a = D  → circle (kappa/kappa_crit = 1 on arc).
        a > D  → sub-critical curvature, axial surfaces spread out.
        """
        alpha = np.linspace(np.pi, 1.5 * np.pi, n_arc)
        x_arc = a + a * np.cos(alpha)
        z_arc = D * np.sin(alpha)
        # tangent angle
        dx_da = -a * np.sin(alpha)
        dz_da = D * np.cos(alpha)
        th_arc = np.arctan2(dz_da, dx_da)
        # curvature of ellipse: kappa = aD / (a²sin²α + D²cos²α)^(3/2)
        denom = (a**2 * np.sin(alpha)**2 + D**2 * np.cos(alpha)**2) ** 1.5
        k_arc = a * D / denom

        x_flat = np.linspace(a, x_end, n_flat + 1)[1:]
        z_flat = np.full(n_flat, -D)
        th_flat = np.zeros(n_flat)
        k_flat = np.zeros(n_flat)

        x = np.concatenate([x_arc, x_flat])
        z = np.concatenate([z_arc, z_flat])
        th = np.concatenate([th_arc, th_flat])
        k = np.concatenate([k_arc, k_flat])
        return SmoothFault(x, z, th, k)

    # ── query interface ───────────────────────────────────────────

    @property
    def arc_length(self):
        return self.L

    def evaluate(self, xi):
        xi = np.asarray(xi, float)
        return (np.interp(xi, self._s, self._x),
                np.interp(xi, self._s, self._z))

    def theta(self, xi):
        return np.interp(np.asarray(xi, float), self._s, self._theta)

    def curvature(self, xi):
        return np.interp(np.asarray(xi, float), self._s, self._kappa)

    def kappa_crit(self, xi):
        _, z = self.evaluate(xi)
        th = self.theta(xi)
        return np.abs(np.cos(th)) / np.maximum(np.abs(z), 0.1)

    def max_violation(self, z_min=1.0):
        """Max |kappa|/kappa_crit, excluding points shallower than z_min."""
        kc = self.kappa_crit(self._s)
        ratio = np.abs(self._kappa) / np.maximum(kc, 1e-16)
        deep = np.abs(self._z) >= z_min
        if not np.any(deep):
            return 0.0
        return float(np.max(ratio[deep]))

    def axial_surface_x(self, xi):
        xi = np.asarray(xi, float)
        xf, zf = self.evaluate(xi)
        th = self.theta(xi)
        ct = np.cos(th)
        t = np.where(np.abs(ct) > 1e-14, -zf / ct, 0.0)
        return xf + t * (-np.sin(th))


# ═══════════════════════════════════════════════════════════════════
# Continuous velocity via Gauss-Legendre quadrature
# ═══════════════════════════════════════════════════════════════════

def continuous_velocity(x_obs, fault, slip_func, n_quad=200, xi_range=None):
    x_obs = np.asarray(x_obs, float)
    L = fault.arc_length
    if xi_range is None:
        a, b = 0.0, L
    else:
        a, b = xi_range
    nodes, weights = np.polynomial.legendre.leggauss(n_quad)
    xi_q = 0.5 * (b - a) * (nodes + 1.0) + a
    w_q = 0.5 * (b - a) * weights

    xf, zf = fault.evaluate(xi_q)
    th = fault.theta(xi_q)
    kappa = fault.curvature(xi_q)
    slip = np.asarray(slip_func(xi_q), float)

    ct, st = np.cos(th), np.sin(th)
    t_ax = np.where(np.abs(ct) > 1e-14, -zf / ct, 0.0)
    x_ax = xf + t_ax * (-st)

    U1 = np.zeros_like(x_obs)
    U2 = np.zeros_like(x_obs)
    eps = 0.01

    for q in range(n_quad):
        sq = slip[q]
        if abs(sq) < 1e-16:
            continue
        wt = w_q[q]

        # Fault element
        x1f = xf[q] - eps * ct[q]
        z1f = zf[q] - eps * st[q]
        x2f = xf[q] + eps * ct[q]
        z2f = zf[q] + eps * st[q]
        sc = wt / (2.0 * eps)
        U1 += sc * u1_segment(x_obs, x1f, z1f, x2f, z2f, -sq)
        U2 += sc * u2_segment(x_obs, x1f, z1f, x2f, z2f, -sq)

        # Fold (axial surface) element
        kq = kappa[q]
        if abs(kq) < 1e-16:
            continue
        fs = sq * kq * wt
        U1 += u1_segment(x_obs, x_ax[q], 0.0, xf[q], zf[q], +fs)
        U2 += u2_segment(x_obs, x_ax[q], 0.0, xf[q], zf[q], -fs)

    return U1, U2


def structural_velocity(x_obs, fault, slip_rate):
    x_obs = np.asarray(x_obs, float)
    xi_d = np.linspace(0, fault.arc_length, 5000)
    xs_d = fault.axial_surface_x(xi_d)
    xi_star = np.interp(x_obs, xs_d, xi_d, left=np.nan, right=xi_d[-1])
    th = fault.theta(np.nan_to_num(xi_star, nan=0.0))
    u = slip_rate * np.cos(th)
    v = -slip_rate * np.sin(th)
    xtip, _ = fault.evaluate(0.0)
    mask = np.isnan(xi_star) | (x_obs < xtip)
    u[mask] = 0.0;  v[mask] = 0.0
    return u, v


# ═══════════════════════════════════════════════════════════════════
# Discrete infrastructure (for validation)
# ═══════════════════════════════════════════════════════════════════

def _seg_angles(xn, zn):
    return np.arctan2(np.diff(zn), np.diff(xn))


def _axial_x(xn, zn, th):
    xa = np.empty(len(th) - 1)
    for j in range(1, len(th)):
        g = 0.5 * (th[j-1] + th[j] + np.pi)
        tg = np.tan(g)
        xa[j-1] = (xn[j] - zn[j] / tg) if abs(tg) > 1e-12 else np.nan
    return xa


def build_discrete(xn, zn, slip_f):
    th = _seg_angles(xn, zn)
    xa = _axial_x(xn, zn, th)
    ns = len(th)
    slip_f = np.broadcast_to(np.asarray(slip_f, float), ns).copy()
    segs = []
    for i in range(ns):
        segs.append(dict(x1=xn[i], z1=zn[i], x2=xn[i+1], z2=zn[i+1],
                         su1=-slip_f[i], su2=-slip_f[i]))
    for j in range(1, ns):
        if not np.isfinite(xa[j-1]):
            continue
        dt = th[j] - th[j-1]
        sl = 0.5 * (slip_f[j-1] + slip_f[j])
        sf = 2.0 * sl * np.sin(0.5 * dt)
        segs.append(dict(x1=xa[j-1], z1=0.0, x2=xn[j], z2=zn[j],
                         su1=+sf, su2=-sf))
    return segs


def elastic_discrete(x_obs, segs):
    U1 = np.zeros_like(x_obs)
    U2 = np.zeros_like(x_obs)
    for sg in segs:
        U1 += u1_segment(x_obs, sg['x1'], sg['z1'], sg['x2'], sg['z2'], sg['su1'])
        U2 += u2_segment(x_obs, sg['x1'], sg['z1'], sg['x2'], sg['z2'], sg['su2'])
    return U1, U2


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    radii = [10, 20, 30]       # km — three circle sizes
    x_plot = 100.0             # km, observation / plot range
    x_fault = 300.0            # km, fault extent (avoids endpoint strain)
    s0 = 20.0                  # mm/yr slip rate
    n_q = 500                  # quadrature points

    # ── Build faults ───────────────────────────────────────────────
    faults = {}
    for R in radii:
        f = SmoothFault.circle_flat(R, x_end=x_fault)
        x_lock = float(R)      # flat starts at x = R
        print(f"R={R:3d} km: L={f.arc_length:.1f} km, "
              f"max |k|/k_crit = {f.max_violation():.3f}, "
              f"lock at ({R}, {-R}) km")
        faults[R] = dict(fault=f, x_lock=x_lock)

    # ── Diagnostic: geometry + kappa/kappa_crit ────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(5, 5), sharex=True,
                              gridspec_kw={'height_ratios': [3, 1]})
    fig.subplots_adjust(hspace=0.3)
    colors = ['steelblue', 'firebrick', '#228833']

    for (R, info), col in zip(faults.items(), colors):
        f = info['fault']
        xi = np.linspace(0, f.arc_length, 3000)
        xp, zp = f.evaluate(xi)
        axes[0].plot(xp, zp, color=col, lw=1.5, label=f'R = {R} km')

        # kappa / kappa_crit vs x (not xi)
        kk = np.abs(f.curvature(xi))
        kc = f.kappa_crit(xi)
        ratio = kk / np.maximum(kc, 1e-16)
        axes[1].plot(xp, ratio, color=col, lw=1.2, label=f'R = {R} km')

        # Draw the circle (dashed) and center
        th_c = np.linspace(0, 2 * np.pi, 200)
        axes[0].plot(R + R * np.cos(th_c), R * np.sin(th_c),
                     color=col, lw=0.5, ls='--', alpha=0.4)
        axes[0].plot(R, 0, 'o', color=col, ms=3, alpha=0.6)

    axes[0].axhline(0, color='k', lw=0.8)
    axes[0].set_ylabel('Depth (km)')
    axes[0].set_title('Fault geometry (circle + flat)')
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].set_xlim(0, x_plot);  axes[0].set_ylim(-50, 10)

    axes[1].axhline(1.0, color='grey', lw=0.8, ls='--',
                     label=r'$|\kappa|/\kappa_{\rm crit} = 1$')
    axes[1].set_ylabel(r'$|\kappa| / \kappa_{\rm crit}$')
    axes[1].set_xlabel('x (km)')
    axes[1].set_title('Curvature ratio (= 1 on arc, 0 on flat)')
    axes[1].legend(fontsize=7, frameon=False, ncol=2)
    axes[1].set_ylim(-0.1, 1.5)
    plt.savefig("diagnostic.pdf", bbox_inches="tight")
    plt.show()

    # ── Earthquake cycle for each radius ───────────────────────────
    x_obs = np.linspace(2.0, x_plot, 500)
    slip_full = lambda xi: np.full_like(np.asarray(xi, float), s0)

    fig, axes = plt.subplots(2, len(radii), figsize=(8, 4),
                              sharex=True, sharey='row',
                              gridspec_kw={"height_ratios": [1, 1],
                                           "hspace": 0.15, "wspace": 0.12},
                              layout="constrained")
    for col, (R, info) in enumerate(faults.items()):
        f = info['fault'];  xl = info['x_lock']

        # Find xi at lock
        xi_d = np.linspace(0, f.arc_length, 5000)
        xd, _ = f.evaluate(xi_d)
        xi_lock = float(np.interp(xl, xd, xi_d))

        # Structural: elastic full slip, split arc + flat for resolution
        _, v_arc = continuous_velocity(x_obs, f, slip_full,
                                       n_quad=n_q, xi_range=(0, xi_lock))
        _, v_flat = continuous_velocity(x_obs, f, slip_full,
                                        n_quad=n_q, xi_range=(xi_lock, f.arc_length))
        v_str = v_arc + v_flat
        # Coseismic: elastic with slip only on locked arc
        U2_co = v_arc  # same integral — locked region is the arc
        inter_v = v_str - U2_co

        faults[R]['v_str'] = v_str
        faults[R]['U2_co'] = U2_co
        faults[R]['inter_v'] = inter_v
        faults[R]['xi_lock'] = xi_lock

        # Top: velocities
        ax_t = axes[0, col]
        ax_t.plot(x_obs, v_str,   'k',         lw=1,   label='Structural')
        ax_t.plot(x_obs, U2_co,   'steelblue', lw=0.9, label='Coseismic')
        ax_t.plot(x_obs, inter_v, 'firebrick', lw=0.9, label='Interseismic')
        ax_t.axhline(0, color='grey', lw=0.5, ls='--')
        ax_t.axvline(xl, color='grey', lw=0.5, ls=':', alpha=0.5)
        ax_t.set_title(f'R = {R} km')
        if col == 0:
            ax_t.set_ylabel(r'$\dot v$ (mm/yr)')
            ax_t.legend(fontsize=6, frameon=False, loc='upper right')

        # Bottom: fault geometry
        ax_b = axes[1, col]
        xi_a = np.linspace(0, f.arc_length, 1000)
        xfa, zfa = f.evaluate(xi_a)
        ax_b.plot(xfa, zfa, 'k', lw=1.5)
        ax_b.plot(*f.evaluate(xi_lock), 'ro', ms=4, zorder=5)
        ax_b.axhline(0, color='k', lw=0.8)
        ax_b.set_xlim(0, x_plot);  ax_b.set_ylim(-60, 5)
        ax_b.set_xlabel('Distance (km)')
        if col == 0:
            ax_b.set_ylabel('Depth (km)')

    plt.savefig("earthquake_cycle.pdf", bbox_inches="tight")
    plt.show()

    # ── Validation: discrete convergence for R = 25 ───────────────
    R_val = 20
    fv = faults[R_val];  tf = fv['fault'];  xl = fv['x_lock']
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(5, 4), sharex=True,
                                  layout="constrained")
    a1.plot(x_obs, fv['U2_co'], 'k', lw=2, label='Continuous')
    a2.plot(x_obs, fv['inter_v'], 'k', lw=2, label='Continuous')
    clrs = plt.cm.viridis(np.linspace(0.2, 0.9, 4))
    for ni, cl in zip([5, 11, 51, 201], clrs):
        xn, zn = tf.evaluate(np.linspace(0, tf.arc_length, ni))
        ns = ni - 1
        xm = 0.5 * (xn[:-1] + xn[1:])
        sl = np.where(xm < xl, s0, 0.0)
        _, u2c = elastic_discrete(x_obs, build_discrete(xn, zn, sl))
        _, vs = continuous_velocity(x_obs, tf, slip_full, n_quad=n_q)
        a1.plot(x_obs, u2c, color=cl, lw=0.8, alpha=0.8, label=f'N={ns}')
        a2.plot(x_obs, vs - u2c, color=cl, lw=0.8, alpha=0.8, label=f'N={ns}')
    for a in (a1, a2):
        a.axhline(0, color='grey', lw=0.3)
        a.axvline(xl, color='grey', ls=':', lw=0.5)
        a.legend(fontsize=6, frameon=False, loc='upper right')
    a1.set_ylabel('Coseismic (mm/yr)')
    a1.set_title(f'Discrete convergence (R = {R_val} km)')
    a2.set_ylabel('Interseismic (mm/yr)');  a2.set_xlabel('Distance (km)')
    plt.savefig("validation.pdf", bbox_inches="tight")
    plt.show()

    print("Done.")
