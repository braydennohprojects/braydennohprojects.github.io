#!/usr/bin/env python3
"""C2 quintic-patch fault geometry with Gauss-Legendre quadrature.

Fault geometry is defined by control points smoothed with C2 quintic
patches at each bend.  The elastic velocity field is computed via
Gauss-Legendre quadrature over the continuous fault + fold dislocation.

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

# ===================================================================
# C2 fault geometry (quintic patch smoothing)
# ===================================================================

def _quintic_patch_coeffs(xL, zL, mL, xR, zR, mR):
    h = xR - xL
    if h <= 0:
        raise ValueError("xR must be > xL")
    a0 = zL
    a1 = mL * h
    a2 = 0.0
    D0 = zR - zL - a1
    D1 = (mR - mL) * h
    A = np.array([[1,  1,  1],
                  [3,  4,  5],
                  [6, 12, 20]], dtype=float)
    a3, a4, a5 = np.linalg.solve(A, np.array([D0, D1, 0.0], dtype=float))
    return np.array([a0, a1, a2, a3, a4, a5], dtype=float), h


def _eval_quintic(x, xL, coeffs, h):
    t = (x - xL) / h
    a0, a1, a2, a3, a4, a5 = coeffs
    return (((((a5*t + a4)*t + a3)*t + a2)*t + a1)*t + a0)


def build_c2_fault_model(x, z, w=5.0):
    x = np.asarray(x, float)
    z = np.asarray(z, float)
    if np.any(np.diff(x) <= 0):
        raise ValueError("x must be strictly increasing")
    n = len(x)
    if n < 2:
        raise ValueError("Need at least 2 points")
    m_seg = np.diff(z) / np.diff(x)
    wv = np.zeros(n)
    for i in range(1, n - 1):
        wi = float(w)
        wi = min(wi, 0.45 * (x[i] - x[i - 1]), 0.45 * (x[i + 1] - x[i]))
        wv[i] = wi
    patches = {}
    for i in range(1, n - 1):
        wi = wv[i]
        if wi <= 0:
            continue
        xL = x[i] - wi
        xR = x[i] + wi
        mL = m_seg[i - 1]
        mR = m_seg[i]
        zL = z[i] + mL * (xL - x[i])
        zR = z[i] + mR * (xR - x[i])
        coeffs, h = _quintic_patch_coeffs(xL, zL, mL, xR, zR, mR)
        patches[i] = (xL, coeffs, h)
    pieces = []
    for seg in range(n - 1):
        left_cut = wv[seg] if seg > 0 else 0.0
        right_cut = wv[seg + 1] if (seg + 1) < n - 1 else 0.0
        xL = x[seg] + left_cut
        xR = x[seg + 1] - right_cut
        if xR > xL:
            pieces.append(("line", xL, xR, seg))
        v = seg + 1
        if 1 <= v <= n - 2 and wv[v] > 0:
            pieces.append(("patch", x[v] - wv[v], x[v] + wv[v], v))
    return dict(x=x, z=z, m_seg=m_seg, pieces=pieces, patches=patches)


def eval_c2_fault(xq, model, extrapolate=True):
    x = model["x"]
    z = model["z"]
    m_seg = model["m_seg"]
    pieces = model["pieces"]
    patches = model["patches"]
    xq = np.asarray(xq, float)
    zq = np.full_like(xq, np.nan, dtype=float)
    for kind, xL, xR, idx in pieces:
        mask = (xq >= xL) & (xq <= xR)
        if not np.any(mask):
            continue
        if kind == "line":
            seg = idx
            zq[mask] = z[seg] + m_seg[seg] * (xq[mask] - x[seg])
        else:
            v = idx
            xL0, coeffs, h = patches[v]
            zq[mask] = _eval_quintic(xq[mask], xL0, coeffs, h)
    if extrapolate:
        m0 = m_seg[0]
        mn = m_seg[-1]
        zq = np.where(np.isnan(zq) & (xq < x[0]),
                       z[0] + m0 * (xq - x[0]), zq)
        zq = np.where(np.isnan(zq) & (xq > x[-1]),
                       z[-1] + mn * (xq - x[-1]), zq)
    return zq


def eval_c2_derivatives(xq, model):
    """Analytically compute z'(x) and z''(x) from the C2 model."""
    x = model["x"]
    z = model["z"]
    m_seg = model["m_seg"]
    pieces = model["pieces"]
    patches = model["patches"]
    xq = np.asarray(xq, float)
    zp = np.full_like(xq, np.nan, dtype=float)
    zpp = np.full_like(xq, np.nan, dtype=float)
    for kind, xL, xR, idx in pieces:
        mask = (xq >= xL) & (xq <= xR)
        if not np.any(mask):
            continue
        if kind == "line":
            zp[mask] = m_seg[idx]
            zpp[mask] = 0.0
        else:
            xL0, coeffs, h = patches[idx]
            a0, a1, a2, a3, a4, a5 = coeffs
            t = (xq[mask] - xL0) / h
            zp[mask] = (a1 + t*(2*a2 + t*(3*a3 + t*(4*a4 + t*5*a5)))) / h
            zpp[mask] = (2*a2 + t*(6*a3 + t*(12*a4 + t*20*a5))) / h**2
    # Extrapolate
    m0, mn = m_seg[0], m_seg[-1]
    left = np.isnan(zp) & (xq < x[0])
    right = np.isnan(zp) & (xq > x[-1])
    zp[left] = m0;   zpp[left] = 0.0
    zp[right] = mn;  zpp[right] = 0.0
    return zp, zpp


def resample_equal_arclength(x_dense, z_dense, n_points):
    x_dense = np.asarray(x_dense, float)
    z_dense = np.asarray(z_dense, float)
    ds = np.sqrt(np.diff(x_dense)**2 + np.diff(z_dense)**2)
    s = np.concatenate(([0.0], np.cumsum(ds)))
    L = s[-1]
    s_uniform = np.linspace(0.0, L, n_points)
    x_new = np.interp(s_uniform, s, x_dense)
    z_new = np.interp(s_uniform, s, z_dense)
    return x_new, z_new, s_uniform, L


# ===================================================================
# Edge dislocation kernels
# ===================================================================

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


# ===================================================================
# SmoothFault
# ===================================================================

class SmoothFault:
    """Fault with numerically defined (x, z, theta, kappa)."""

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
    def from_c2(xctrl, zctrl, w=2.0, x_end=200.0,
                n_dense=5000, n_resample=3000):
        """Build smooth fault from C2 quintic-patch geometry.

        Parameters
        ----------
        xctrl, zctrl : array-like
            Control point coordinates.
        w : float
            Half-width of quintic blending patches.
        x_end : float
            Extend fault horizontally to this x.
        n_dense : int
            Dense evaluation grid size.
        n_resample : int
            Number of equal-arc-length resampled points.
        """
        xc = list(xctrl)
        zc = list(zctrl)
        # Extend last segment slope before going flat, so the
        # ramp-to-flat C2 patch stays outside the user's domain.
        if x_end > xc[-1] and len(xc) >= 2:
            m_last = (zc[-1] - zc[-2]) / (xc[-1] - xc[-2])
            x_buf = xc[-1] + 3 * w   # buffer = 3 patch widths
            z_buf = zc[-1] + m_last * (x_buf - xc[-1])
            xc.extend([x_buf, x_end])
            zc.extend([z_buf, z_buf])
        model = build_c2_fault_model(np.array(xc), np.array(zc), w=w)
        x_dense = np.linspace(xc[0], xc[-1], n_dense)
        z_dense = eval_c2_fault(x_dense, model)
        zp, zpp = eval_c2_derivatives(x_dense, model)

        # Analytical theta and kappa from z'(x) and z''(x)
        theta_dense = np.arctan2(zp, np.ones_like(zp))
        kappa_dense = zpp / (1.0 + zp**2)**1.5

        # Resample to equal arc length
        x_rs, z_rs, s_rs, _ = resample_equal_arclength(
            x_dense, z_dense, n_resample)

        # Interpolate analytical theta and kappa onto resampled grid
        ds = np.sqrt(np.diff(x_dense)**2 + np.diff(z_dense)**2)
        s_dense = np.concatenate(([0.0], np.cumsum(ds)))
        theta_rs = np.interp(s_rs, s_dense, theta_dense)
        kappa_rs = np.interp(s_rs, s_dense, kappa_dense)

        return SmoothFault(x_rs, z_rs, theta_rs, kappa_rs)

    # -- query interface --

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

    def axial_surface_x(self, xi):
        xi = np.asarray(xi, float)
        xf, zf = self.evaluate(xi)
        th = self.theta(xi)
        ct = np.cos(th)
        t = np.where(np.abs(ct) > 1e-14, -zf / ct, 0.0)
        return xf + t * (-np.sin(th))


# ===================================================================
# Continuous velocity via Gauss-Legendre quadrature
# ===================================================================

def _continuous_velocity_single(x_obs, fault, slip_func, n_quad, a, b):
    """Single-panel Gauss-Legendre quadrature over [a, b]."""
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


def continuous_velocity(x_obs, fault, slip_func, n_quad=200,
                        xi_range=None, n_panels=20):
    """Composite Gauss-Legendre quadrature.

    Splits the integration range into n_panels equal sub-intervals,
    each with n_quad/n_panels quadrature points.  This resolves
    narrow curvature peaks that a single panel would under-sample.
    """
    x_obs = np.asarray(x_obs, float)
    L = fault.arc_length
    if xi_range is None:
        a, b = 0.0, L
    else:
        a, b = xi_range
    edges = np.linspace(a, b, n_panels + 1)
    n_per = max(n_quad // n_panels, 20)
    U1 = np.zeros_like(x_obs)
    U2 = np.zeros_like(x_obs)
    for i in range(n_panels):
        u1, u2 = _continuous_velocity_single(
            x_obs, fault, slip_func, n_per, edges[i], edges[i + 1])
        U1 += u1
        U2 += u2
    return U1, U2


def structural_velocity(x_obs, fault, slip_rate):
    x_obs = np.asarray(x_obs, float)
    xi_d = np.linspace(0, fault.arc_length, 10000)
    xs_d = fault.axial_surface_x(xi_d)
    xi_star = np.interp(x_obs, xs_d, xi_d, left=np.nan, right=xi_d[-1])
    th = fault.theta(np.nan_to_num(xi_star, nan=0.0))
    u = slip_rate * np.cos(th)
    v = -slip_rate * np.sin(th)
    xtip, _ = fault.evaluate(0.0)
    mask = np.isnan(xi_star) | (x_obs < xtip)
    u[mask] = 0.0
    v[mask] = 0.0
    return u, v


CALTECH_ORANGE = '#FF6C0C'


def particle_path(z0, f, x_start=200.0):
    """Integrate a particle from (x_start, z0) leftward through the fold.

    Velocity at each step is set by the nearest axial surface to the
    RIGHT of the particle (the most recently crossed).
    """
    if z0 >= -0.01:
        return np.array([x_start, -5]), np.array([0.0, 0.0])

    xi_dense = np.linspace(f.arc_length * 0.99, 0.001, 2000)
    xf_all, zf_all = f.evaluate(xi_dense)
    th_all = f.theta(xi_dense)

    x_curr, z_curr = x_start, z0
    path_x = [x_curr]
    path_z = [z_curr]
    dx = -0.25

    for _ in range(5000):
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

        to_right = x_ax >= x_curr
        if np.any(to_right):
            idx_r = np.where(to_right)[0]
            i_active = idx_r[np.argmin(x_ax[to_right])]
            theta = float(th[i_active])
        else:
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


# ===================================================================
# Main
# ===================================================================

if __name__ == "__main__":

    s0 = 1.0           # unit slip rate
    x_lock = 120.0      # locking at x = 120
    n_q = 10000         # quadrature points (total across all panels)
    n_pan = 200         # composite panels (ensures many panels per curvature zone)
    x_plot = 200.0      # plotting range

    # C2 fault control points
    xctrl = [0, 6, 83, 112, 200]
    zctrl = [-0.01, -3, -12, -21, -29]
    w_blend = 8.0

    print("Building C2 fault from control points:")
    for xc, zc in zip(xctrl, zctrl):
        print(f"  ({xc}, {zc})")

    f = SmoothFault.from_c2(xctrl, zctrl, w=w_blend, x_end=300.0)
    print(f"Fault arc length: {f.arc_length:.2f}")

    # Find xi at locking position
    xi_d = np.linspace(0, f.arc_length, 10000)
    xd, _ = f.evaluate(xi_d)
    xi_lock = float(np.interp(x_lock, xd, xi_d))
    xf_lock, zf_lock = f.evaluate(xi_lock)
    print(f"Lock point: xi = {xi_lock:.3f}, "
          f"(x, z) = ({xf_lock:.3f}, {zf_lock:.3f})")

    # Observation points
    x_obs = np.linspace(0.3, x_plot, 600)
    slip_full = lambda xi: np.full_like(np.asarray(xi, float), s0)

    # Structural velocity
    u_kin, v_kin = structural_velocity(x_obs, f, s0)

    # Coseismic: elastic integral over locked portion only
    u_co, v_co = continuous_velocity(x_obs, f, slip_full,
                                     n_quad=n_q, xi_range=(0, xi_lock),
                                     n_panels=n_pan)

    # Interseismic = structural - coseismic
    v_inter = v_kin - v_co
    u_inter = u_kin - u_co

    # ---- Diagnostic: spatial gradient ----
    dx = np.diff(x_obs)
    grad_co = np.diff(v_co) / dx
    grad_inter = np.diff(v_inter) / dx
    x_mid = 0.5 * (x_obs[:-1] + x_obs[1:])

    fig_diag, ax_diag = plt.subplots(1, 1, figsize=(7, 3), layout="constrained")
    ax_diag.plot(x_mid, grad_co, 'steelblue', lw=0.8, label='dV_co/dx')
    ax_diag.plot(x_mid, grad_inter, 'firebrick', lw=0.8, label='dV_inter/dx')
    ax_diag.axhline(0, color='k', lw=0.3)
    ax_diag.set_xlabel('Distance')
    ax_diag.set_ylabel('Spatial gradient')
    ax_diag.legend(fontsize=7, frameon=False)
    ax_diag.set_title('Spatial gradient of coseismic & interseismic')
    ax_diag.set_xlim(0, x_plot)
    fig_diag.savefig("gradient_diagnostic.pdf", bbox_inches="tight")

    # ---- Particle paths for layering ----
    xi_right = float(np.interp(x_plot, xd, xi_d))
    _, z_right = f.evaluate(xi_right)
    z_particles = np.linspace(-0.5, float(z_right), 21)

    print("Computing particle paths...")
    paths = []
    for z0 in z_particles:
        xp, zp = particle_path(z0, f, x_start=x_plot)
        paths.append((xp, zp))

    x_grid = np.linspace(-5, x_plot, 800)
    z_on_grid = []
    for xp, zp in paths:
        xp_inc = xp[::-1]
        zp_inc = zp[::-1]
        z_interp = np.interp(x_grid, xp_inc, zp_inc,
                             left=0.0, right=zp_inc[-1])
        z_interp = np.minimum(z_interp, 0.0)
        z_on_grid.append(z_interp)

    # ---- Plotting ----
    fig, axes = plt.subplots(3, 1, figsize=(6, 5),
                              gridspec_kw={"height_ratios": [1, 1, 1.2],
                                           "hspace": 0.05},
                              sharex=True,
                              layout="constrained")

    # Top: vertical velocity
    ax_v = axes[0]
    ax_v.plot(x_obs, v_kin, 'k', lw=1.2, label='Structural')
    ax_v.plot(x_obs, v_co, 'steelblue', lw=1.2, label='Coseismic')
    ax_v.plot(x_obs, v_inter, 'firebrick', lw=1.2, label='Interseismic')
    ax_v.set_ylabel(r'$v_z\;/\;s$')
    ax_v.legend(fontsize=7, frameon=False, loc='lower left')
    ax_v.set_title('Vertical velocity')

    # Middle: horizontal velocity
    ax_h = axes[1]
    ax_h.plot(x_obs, u_kin, 'k', lw=1.2, label='Structural')
    ax_h.plot(x_obs, u_co, 'steelblue', lw=1.2, label='Coseismic')
    ax_h.plot(x_obs, u_inter, 'firebrick', lw=1.2, label='Interseismic')
    ax_h.set_ylabel(r'$v_x\;/\;s$')
    ax_h.legend(fontsize=7, frameon=False, loc='lower left')
    ax_h.set_title('Horizontal velocity')

    # Bottom: fault geometry with particle layering
    ax_g = axes[2]

    # Alternating fills
    for i in range(len(z_on_grid) - 1):
        color = CALTECH_ORANGE if i % 2 == 0 else 'white'
        ax_g.fill_between(x_grid, z_on_grid[i], z_on_grid[i + 1],
                          color=color, alpha=1.0, zorder=1, edgecolor='none')

    # Particle boundary lines
    for xp, zp in paths:
        ax_g.plot(xp, zp, 'k', lw=0.4, zorder=3)

    # Draw axial surfaces
    n_ax = 400
    xi_ax_samples = np.linspace(0.05, f.arc_length * 0.95, n_ax)
    for xi_i in xi_ax_samples:
        xfi = float(f.evaluate(xi_i)[0])
        zfi = float(f.evaluate(xi_i)[1])
        th_i = float(f.theta(xi_i))
        ct = np.cos(th_i)
        if abs(ct) < 1e-10:
            continue
        t = -zfi / ct
        if t < 0:
            continue
        x_s = xfi + t * (-np.sin(th_i))
        if -5 <= x_s <= x_plot + 5:
            ax_g.plot([xfi, x_s], [zfi, 0], color='grey',
                      lw=0.3, alpha=0.3, zorder=2)

    # Draw fault: locked (blue) and creeping (red)
    xi_a = np.linspace(0, f.arc_length, 5000)
    xfa, zfa = f.evaluate(xi_a)
    locked_mask = xi_a <= xi_lock
    creep_mask = xi_a >= xi_lock
    ax_g.plot(xfa[locked_mask], zfa[locked_mask],
              color='steelblue', lw=1.2, zorder=4, label='Locked')
    ax_g.plot(xfa[creep_mask], zfa[creep_mask],
              color='firebrick', lw=1.2, zorder=4, label='Creeping')

    # Mark locking point
    ax_g.plot(xf_lock, zf_lock, 'ko', ms=2.5, zorder=5,
              label='Locking point')

    ax_g.axhline(0, color='k', lw=0.8, zorder=5)
    ax_g.set_xlim(0, x_plot)
    ax_g.set_ylim(-35, 5)
    ax_g.set_xlabel('Distance')
    ax_g.set_ylabel('Depth')
    ax_g.legend(fontsize=7, frameon=False, loc='lower left')
    ax_g.set_title('Fault geometry')

    plt.savefig("staircase_fault.pdf", bbox_inches="tight")
    plt.show()
    print("Done.")
