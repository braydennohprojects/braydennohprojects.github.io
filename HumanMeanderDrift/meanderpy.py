"""
Minimal curvature-driven meandering model with cutoff detection.

Provides four core functions:
  _migrate()                    — Numba-JIT fused migration kernel
  _resample()                   — Numba-JIT linear resampling
  _smooth()                     — Numba-JIT 3-point smoothing
  detect_and_execute_cutoffs()  — cKDTree-based neck cutoff detection

Plus helpers:
  make_initial_channel()        — sinusoidal initial centerline
"""

import numpy as np
import numba
from scipy.spatial import cKDTree


# ---------------------------------------------------------------------------
# JIT-compiled migration kernel
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _migrate(x, y, kl_W, dt, alpha, omega, gamma):
    """Fused: curvature -> upstream convolution -> displacement. O(n).
    Endpoints fixed, all interior nodes migrate freely."""
    n = len(x)
    # -- first derivatives (central, one-sided at edges) --
    dx = np.empty(n)
    dy = np.empty(n)
    dx[0] = x[1] - x[0];          dy[0] = y[1] - y[0]
    dx[n-1] = x[n-1] - x[n-2];    dy[n-1] = y[n-1] - y[n-2]
    for i in range(1, n-1):
        dx[i] = (x[i+1] - x[i-1]) * 0.5
        dy[i] = (y[i+1] - y[i-1]) * 0.5
    # -- second derivatives --
    ddx = np.empty(n)
    ddy = np.empty(n)
    ddx[0] = dx[1] - dx[0];       ddy[0] = dy[1] - dy[0]
    ddx[n-1] = dx[n-1] - dx[n-2]; ddy[n-1] = dy[n-1] - dy[n-2]
    for i in range(1, n-1):
        ddx[i] = (dx[i+1] - dx[i-1]) * 0.5
        ddy[i] = (dy[i+1] - dy[i-1]) * 0.5
    # -- curvature, normal, ds, convolution, displacement in one pass --
    W_sum = 0.0
    S_sum = 0.0
    x_out = x.copy()
    y_out = y.copy()
    for i in range(n):
        denom = (dx[i]**2 + dy[i]**2)**1.5
        if denom < 1e-30: denom = 1e-30
        curv = (dx[i]*ddy[i] - dy[i]*ddx[i]) / denom
        R0 = kl_W * curv
        if i > 0:
            seg = ((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)**0.5
        else:
            seg = 0.0
        decay = np.exp(-alpha * seg)
        W_sum = R0 + decay * W_sum
        S_sum = 1.0 + decay * S_sum
        R = omega * R0 + gamma * W_sum / S_sum
        ds_i = (dx[i]**2 + dy[i]**2)**0.5
        if ds_i < 1e-30: ds_i = 1e-30
        nx = dy[i] / ds_i
        ny = -dx[i] / ds_i
        if i > 0 and i < n - 1:
            x_out[i] = x[i] + R * nx * dt
            y_out[i] = y[i] + R * ny * dt
    return x_out, y_out


@numba.njit(cache=True)
def _migrate_symmetric(x, y, kl_W, dt, alpha, omega, gamma):
    """Symmetric convolution: average of upstream and downstream passes.
    Meanders grow but no downstream phase shift."""
    n = len(x)
    # -- derivatives --
    dx = np.empty(n); dy = np.empty(n)
    dx[0] = x[1]-x[0];       dy[0] = y[1]-y[0]
    dx[n-1] = x[n-1]-x[n-2]; dy[n-1] = y[n-1]-y[n-2]
    for i in range(1, n-1):
        dx[i] = (x[i+1]-x[i-1])*0.5; dy[i] = (y[i+1]-y[i-1])*0.5
    ddx = np.empty(n); ddy = np.empty(n)
    ddx[0] = dx[1]-dx[0];       ddy[0] = dy[1]-dy[0]
    ddx[n-1] = dx[n-1]-dx[n-2]; ddy[n-1] = dy[n-1]-dy[n-2]
    for i in range(1, n-1):
        ddx[i] = (dx[i+1]-dx[i-1])*0.5; ddy[i] = (dy[i+1]-dy[i-1])*0.5

    # -- curvature + R0 for all nodes --
    curv = np.empty(n); R0 = np.empty(n)
    seg_len = np.empty(n)
    for i in range(n):
        denom = (dx[i]**2 + dy[i]**2)**1.5
        if denom < 1e-30: denom = 1e-30
        curv[i] = (dx[i]*ddy[i] - dy[i]*ddx[i]) / denom
        R0[i] = kl_W * curv[i]
        if i > 0:
            seg_len[i] = ((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)**0.5
        else:
            seg_len[i] = 0.0

    # -- forward pass (upstream convolution, i=0..n-1) --
    R_fwd = np.empty(n)
    W_sum = 0.0; S_sum = 0.0
    for i in range(n):
        decay = np.exp(-alpha * seg_len[i])
        W_sum = R0[i] + decay * W_sum
        S_sum = 1.0 + decay * S_sum
        R_fwd[i] = omega * R0[i] + gamma * W_sum / S_sum

    # -- backward pass (downstream convolution, i=n-1..0) --
    R_bwd = np.empty(n)
    W_sum = 0.0; S_sum = 0.0
    for i in range(n-1, -1, -1):
        if i < n-1:
            seg_b = seg_len[i+1]
        else:
            seg_b = 0.0
        decay = np.exp(-alpha * seg_b)
        W_sum = R0[i] + decay * W_sum
        S_sum = 1.0 + decay * S_sum
        R_bwd[i] = omega * R0[i] + gamma * W_sum / S_sum

    # -- average and displace --
    x_out = x.copy(); y_out = y.copy()
    for i in range(n):
        R = 0.5 * (R_fwd[i] + R_bwd[i])
        ds_i = (dx[i]**2 + dy[i]**2)**0.5
        if ds_i < 1e-30: ds_i = 1e-30
        nx = dy[i] / ds_i; ny = -dx[i] / ds_i
        if i > 0 and i < n-1:
            x_out[i] = x[i] + R * nx * dt
            y_out[i] = y[i] + R * ny * dt
    return x_out, y_out


@numba.njit(cache=True)
def _migrate_downstream(x, y, kl_W, dt, alpha, omega, gamma):
    """Same as _migrate but convolution looks downstream (backward pass).
    Phase shift is upstream instead of downstream."""
    n = len(x)
    dx = np.empty(n); dy = np.empty(n)
    dx[0] = x[1]-x[0]; dy[0] = y[1]-y[0]
    dx[n-1] = x[n-1]-x[n-2]; dy[n-1] = y[n-1]-y[n-2]
    for i in range(1, n-1):
        dx[i] = (x[i+1]-x[i-1])*0.5; dy[i] = (y[i+1]-y[i-1])*0.5
    ddx = np.empty(n); ddy = np.empty(n)
    ddx[0] = dx[1]-dx[0]; ddy[0] = dy[1]-dy[0]
    ddx[n-1] = dx[n-1]-dx[n-2]; ddy[n-1] = dy[n-1]-dy[n-2]
    for i in range(1, n-1):
        ddx[i] = (dx[i+1]-dx[i-1])*0.5; ddy[i] = (dy[i+1]-dy[i-1])*0.5
    # curvature + R0
    curv = np.empty(n); R0 = np.empty(n); seg_len = np.empty(n)
    for i in range(n):
        denom = (dx[i]**2 + dy[i]**2)**1.5
        if denom < 1e-30: denom = 1e-30
        curv[i] = (dx[i]*ddy[i] - dy[i]*ddx[i]) / denom
        R0[i] = kl_W * curv[i]
        if i < n-1:
            seg_len[i] = ((x[i+1]-x[i])**2 + (y[i+1]-y[i])**2)**0.5
        else:
            seg_len[i] = 0.0
    # backward pass (downstream convolution, i=n-1..0)
    W_sum = 0.0; S_sum = 0.0
    x_out = x.copy(); y_out = y.copy()
    for i in range(n-1, -1, -1):
        if i < n-1:
            seg_b = seg_len[i]
        else:
            seg_b = 0.0
        decay = np.exp(-alpha * seg_b)
        W_sum = R0[i] + decay * W_sum
        S_sum = 1.0 + decay * S_sum
        R = omega * R0[i] + gamma * W_sum / S_sum
        ds_i = (dx[i]**2 + dy[i]**2)**0.5
        if ds_i < 1e-30: ds_i = 1e-30
        nx = dy[i] / ds_i
        ny = -dx[i] / ds_i
        if i > 0 and i < n-1:
            x_out[i] = x[i] + R * nx * dt
            y_out[i] = y[i] + R * ny * dt
    return x_out, y_out


# ---------------------------------------------------------------------------
# JIT-compiled resample
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _resample(x, y, ds_target):
    """Linear resample preserving endpoints. O(n)."""
    n = len(x)
    s = np.empty(n)
    s[0] = 0.0
    for i in range(1, n):
        s[i] = s[i-1] + ((x[i]-x[i-1])**2 + (y[i]-y[i-1])**2)**0.5
    total = s[n-1]
    if total < 2 * ds_target:
        return x.copy(), y.copy()
    n_new = int(total / ds_target + 0.5)
    if n_new < 2:
        n_new = 2
    x_new = np.empty(n_new + 1)
    y_new = np.empty(n_new + 1)
    x_new[0] = x[0];       y_new[0] = y[0]
    x_new[n_new] = x[n-1]; y_new[n_new] = y[n-1]
    j = 0
    for i in range(1, n_new):
        s_t = total * i / n_new
        while j < n - 2 and s[j+1] < s_t:
            j += 1
        denom = s[j+1] - s[j]
        if denom < 1e-30: denom = 1e-30
        f = (s_t - s[j]) / denom
        x_new[i] = x[j] + f * (x[j+1] - x[j])
        y_new[i] = y[j] + f * (y[j+1] - y[j])
    return x_new, y_new


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------

@numba.njit(cache=True)
def _smooth(x, y, n_passes=3):
    """Light 3-point averaging, endpoints fixed. O(n * n_passes)."""
    for _ in range(n_passes):
        xn = x.copy()
        yn = y.copy()
        for i in range(1, len(x) - 1):
            xn[i] = 0.25 * x[i-1] + 0.5 * x[i] + 0.25 * x[i+1]
            yn[i] = 0.25 * y[i-1] + 0.5 * y[i] + 0.25 * y[i+1]
        x = xn
        y = yn
    return x, y


# ---------------------------------------------------------------------------
# Cutoff detection
# ---------------------------------------------------------------------------

def detect_and_execute_cutoffs(x, y, cutoff_dist, band):
    """Find and execute all neck cutoffs using cKDTree (O(n log n))
    with batch removal of non-overlapping loops."""
    oxbows = []

    while len(x) >= 5:
        tree = cKDTree(np.column_stack((x, y)))
        pairs = tree.query_pairs(cutoff_dist, output_type='ndarray')
        if len(pairs) == 0:
            break

        i_arr = pairs[:, 0].astype(np.intp)
        j_arr = pairs[:, 1].astype(np.intp)
        mask = np.abs(i_arr - j_arr) >= band
        if not np.any(mask):
            break
        i_arr = i_arr[mask]
        j_arr = j_arr[mask]

        swap = i_arr > j_arr
        i_arr[swap], j_arr[swap] = j_arr[swap], i_arr[swap]

        d2 = (x[i_arr] - x[j_arr])**2 + (y[i_arr] - y[j_arr])**2
        order = np.argsort(d2)
        i_arr = i_arr[order]
        j_arr = j_arr[order]

        used = np.zeros(len(x), dtype=np.bool_)
        cuts = []
        for k in range(len(i_arr)):
            ii, jj = i_arr[k], j_arr[k]
            if np.any(used[ii:jj+1]):
                continue
            used[ii:jj+1] = True
            cuts.append((int(ii), int(jj)))

        if not cuts:
            break

        cuts.sort(key=lambda c: c[0], reverse=True)
        for ii, jj in cuts:
            oxbows.append((x[ii:jj+1].copy(), y[ii:jj+1].copy()))
            x = np.concatenate((x[:ii+1], x[jj:]))
            y = np.concatenate((y[:ii+1], y[jj:]))

    return x, y, oxbows


# ---------------------------------------------------------------------------
# Initial channel generator
# ---------------------------------------------------------------------------

def make_initial_channel(length, ds, amplitude=None, wavelength=None,
                         n_bends=None, W=None):
    """Generate a tapered sinusoidal initial centerline."""
    if n_bends is None:
        n_bends = 20
    if wavelength is None:
        wavelength = length / n_bends
    if amplitude is None:
        amplitude = 0.5 * W if W is not None else 0.01 * length
    n = int(round(length / ds)) + 1
    x = np.linspace(0, length, n)
    y = amplitude * np.sin(2 * np.pi * n_bends * x / length)
    taper = np.ones_like(x)
    left = x < wavelength
    right = x > (length - wavelength)
    taper[left] = 0.5 * (1 - np.cos(np.pi * x[left] / wavelength))
    taper[right] = 0.5 * (1 - np.cos(np.pi * (length - x[right]) / wavelength))
    y *= taper
    return x, y
