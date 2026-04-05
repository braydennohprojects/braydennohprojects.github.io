"""
3.26: Production run for PNAS figures.
Symmetric vs Asymmetric with agents.
- PNG every 100 agent frames (high DPI, no compression loss)
- PDF every 1000 agent frames (vector quality)
- npz with drift distributions
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sys, os, time
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.stats import gaussian_kde
from numba import njit
import subprocess

sys.path.insert(0, '/Users/braydennoh/Research/VegetatedRiver/3.14')
import meanderpy as mp

plt.rcParams["font.family"] = "Times New Roman"

BASE = "/Users/braydennoh/Human-River/3.26"
os.makedirs(BASE, exist_ok=True)

# ── River params ────────────────────────────────────────────────────────
W = 10.0
ds = 10.0
length = 15000.0
n_bends = 12

kl = 12.0 / (365.25 * 24 * 3600)
dt = 0.5 * 365.25 * 24 * 3600
kl_W = kl * W
alpha = 0.5 / W   # = 0.05, e-fold = 2W
cutoff_dist = 10.0 * W
band = int(np.ceil(cutoff_dist / ds)) + 3

# ── Grid ────────────────────────────────────────────────────────────────
XMIN, XMAX = 1000.0, 11000.0
YMIN, YMAX = -1000.0, 1000.0
CELL_SIZE = 20.0
COLS = int((XMAX - XMIN) / CELL_SIZE)
ROWS = int((YMAX - YMIN) / CELL_SIZE)

COL_FLOODPLAIN = np.array([0, 0, 0], dtype=np.uint8)       # #000000
COL_RIVER = np.array([51, 101, 255], dtype=np.uint8)        # #3365ff
COL_AGENT = np.array([251, 51, 51], dtype=np.uint8)         # #fb3333

# ── Simulation ──────────────────────────────────────────────────────────
WARMUP = 5000
SIM_STEPS = 100000
TOTAL = WARMUP + SIM_STEPS
SAVE_RIVER_EVERY = 10
N_AGENTS = 1000

# Output intervals (in agent frames, not river steps)
PNG_EVERY = 100
PDF_EVERY = 1000

SCENARIOS = [
    {"name": "Symmetric",  "mode": "symmetric"},
    {"name": "Asymmetric", "mode": "asymmetric"},
    {"name": "Upstream",   "mode": "upstream"},
]


def rasterize(x, y):
    g = np.zeros((ROWS, COLS), dtype=np.bool_)
    xs = np.linspace(x[:-1], x[1:], 100).flatten()
    ys = np.linspace(y[:-1], y[1:], 100).flatten()
    ci = np.floor((xs - XMIN) / CELL_SIZE).astype(int)
    ri = np.floor((ys - YMIN) / CELL_SIZE).astype(int)
    m = (0 <= ci) & (ci < COLS) & (0 <= ri) & (ri < ROWS)
    g[ri[m], ci[m]] = True
    return g


@njit(cache=True)
def step_local(agent_r, agent_c, dist, river, occupied, rows, cols, rand_vals):
    dr = np.array([-1,-1,-1,0,0,1,1,1], dtype=np.int32)
    dc = np.array([-1,0,1,-1,1,-1,0,1], dtype=np.int32)
    N = agent_r.shape[0]
    for i in range(N):
        r, c = agent_r[i], agent_c[i]
        if river[r, c]:
            cr = np.empty(8, dtype=np.int32); cc = np.empty(8, dtype=np.int32); nc = 0
            for k in range(8):
                nr, ncc = r+dr[k], c+dc[k]
                if 0<=nr<rows and 0<=ncc<cols and not river[nr,ncc] and not occupied[nr,ncc]:
                    cr[nc]=nr; cc[nc]=ncc; nc+=1
            if nc > 0:
                p = int(rand_vals[i]*nc)
                if p>=nc: p=nc-1
                occupied[r,c]=False; agent_r[i]=cr[p]; agent_c[i]=cc[p]; occupied[cr[p],cc[p]]=True
        else:
            bd = dist[r,c]
            tr = np.empty(9, dtype=np.int32); tc = np.empty(9, dtype=np.int32)
            tr[0]=r; tc[0]=c; nt=1
            for k in range(8):
                nr, ncc = r+dr[k], c+dc[k]
                if 0<=nr<rows and 0<=ncc<cols and not river[nr,ncc] and not occupied[nr,ncc]:
                    if dist[nr,ncc] < bd:
                        bd=dist[nr,ncc]; nt=0; tr[0]=nr; tc[0]=ncc; nt=1
                    elif dist[nr,ncc] == bd:
                        tr[nt]=nr; tc[nt]=ncc; nt+=1
            p = int(rand_vals[i]*nt)
            if p>=nt: p=nt-1
            nr2, nc2 = tr[p], tc[p]
            if nr2!=r or nc2!=c:
                occupied[r,c]=False; occupied[nr2,nc2]=True
            agent_r[i]=nr2; agent_c[i]=nc2


def render_frame(river, agent_r, agent_c):
    img = np.zeros((ROWS, COLS, 3), dtype=np.uint8)
    img[:,:] = COL_FLOODPLAIN
    img[agent_r, agent_c] = COL_AGENT
    img[river] = COL_RIVER
    return img


# Axis extents in km (x offset to start at 0)
X0_KM = 0.0
X1_KM = (XMAX - XMIN) / 1000.0
Y0_KM = YMIN / 1000.0
Y1_KM = YMAX / 1000.0


def save_fig(out_dir, name, step, river, agent_r, agent_c, fmt="png"):
    img = np.flipud(render_frame(river, agent_r, agent_c))
    fig, ax = plt.subplots(figsize=(6, 1.5))
    ax.imshow(img, extent=[X0_KM, X1_KM, Y0_KM, Y1_KM], aspect='equal',
              interpolation='none', rasterized=True)
    ax.set_xlabel("Along-valley distance (km)")
    ax.set_ylabel("Cross-valley (km)")
    ax.set_xlim(X0_KM, X1_KM); ax.set_ylim(Y0_KM, Y1_KM)
    fig.tight_layout()
    if fmt == "png":
        fig.savefig(os.path.join(out_dir, f"frame_{step:06d}.png"),
                    dpi=300, bbox_inches='tight')
    elif fmt == "pdf":
        fig.savefig(os.path.join(out_dir, f"frame_{step:06d}.pdf"),
                    dpi=600, bbox_inches='tight')
    plt.close(fig)


# ── JIT warmup ──────────────────────────────────────────────────────────
print("JIT warmup...")
x0, y0 = mp.make_initial_channel(length=length, ds=ds, amplitude=W*0.5,
                                  n_bends=n_bends, W=W)
xi, yi = mp._resample(x0[:4].astype(np.float64), y0[:4].astype(np.float64), ds)
mp._migrate(xi, yi, kl_W, dt, alpha, -1.0, 2.5)
mp._migrate_symmetric(xi, yi, kl_W, dt, alpha, -1.0, 2.5)
mp._migrate_downstream(xi, yi, kl_W, dt, alpha, -1.0, 2.5)
mp._smooth(xi, yi, 3)
_r = np.array([10], dtype=np.int32); _c = np.array([10], dtype=np.int32)
_o = np.zeros((ROWS, COLS), dtype=np.bool_)
_d = np.ones((ROWS, COLS), dtype=np.float32)
_rv = np.zeros((ROWS, COLS), dtype=np.bool_)
step_local(_r, _c, _d, _rv, _o, ROWS, COLS, np.array([0.5]))
print("  done\n")

# ── Run all scenarios ──────────────────────────────────────────────────
drift_results = {}

for scen in SCENARIOS:
    name = scen["name"]
    mode = scen["mode"]
    png_dir = os.path.join(BASE, name.lower(), "png")
    pdf_dir = os.path.join(BASE, name.lower(), "pdf")
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(pdf_dir, exist_ok=True)

    # Skip if already run
    existing = [f for f in os.listdir(png_dir) if f.endswith(".png")]
    if len(existing) > 10:
        print(f"=== {name} === SKIPPED (already has {len(existing)} frames)")
        continue

    print(f"=== {name} ({mode}) ===")

    # Phase 1: precompute river masks
    print(f"  [{name}] Precomputing channels...")
    np.random.seed(42)
    x, y = mp.make_initial_channel(length=length, ds=ds, amplitude=W*0.5,
                                    n_bends=n_bends, W=W)
    y[1:-1] += np.random.uniform(-1, 1, size=len(y)-2) * 0.05 * W
    x, y = mp._resample(x.astype(np.float64), y.astype(np.float64), ds)

    river_masks = []
    t0 = time.time()
    n_cutoffs = 0
    for step in range(1, TOTAL + 1):
        if mode == "symmetric":
            x, y = mp._migrate_symmetric(x, y, kl_W, dt, alpha, -1.0, 2.5)
        elif mode == "asymmetric":
            x, y = mp._migrate(x, y, kl_W, dt, alpha, -1.0, 2.5)
        elif mode == "upstream":
            x, y = mp._migrate_downstream(x, y, kl_W, dt, alpha, -1.0, 2.5)
        if step % 5 == 0:
            x, y, oxbows = mp.detect_and_execute_cutoffs(x, y, cutoff_dist, band)
            n_cutoffs += len(oxbows)
            if oxbows:
                x, y = mp._smooth(x, y, 3)
        x, y = mp._resample(x, y, ds)
        if step > WARMUP and (step - WARMUP) % SAVE_RIVER_EVERY == 0:
            river_masks.append(rasterize(x, y))

    river_masks = np.array(river_masks, dtype=np.bool_)
    N_FRAMES = river_masks.shape[0]
    print(f"  [{name}] Channels done in {time.time()-t0:.1f}s, "
          f"{n_cutoffs} cutoffs, {N_FRAMES} frames")

    # Phase 2: run agents
    print(f"  [{name}] Running agents...")
    rng = np.random.default_rng(42)
    valid = np.argwhere(~river_masks[0])
    chosen = rng.choice(len(valid), size=N_AGENTS, replace=False)
    agent_r = valid[chosen, 0].astype(np.int32).copy()
    agent_c = valid[chosen, 1].astype(np.int32).copy()
    occupied = np.zeros((ROWS, COLS), dtype=np.bool_)
    occupied[agent_r, agent_c] = True
    init_c = agent_c.copy()

    # Save initial (both PNG and PDF)
    save_fig(png_dir, name, 0, river_masks[0], agent_r, agent_c, "png")
    save_fig(pdf_dir, name, 0, river_masks[0], agent_r, agent_c, "pdf")

    t0 = time.time()
    for i in range(N_FRAMES):
        dist = distance_transform_edt(~river_masks[i]).astype(np.float32)
        rv = rng.random(N_AGENTS)
        step_local(agent_r, agent_c, dist, river_masks[i], occupied, ROWS, COLS, rv)

        frame_num = i + 1
        if frame_num % PNG_EVERY == 0:
            save_fig(png_dir, name, frame_num, river_masks[i], agent_r, agent_c, "png")
        if frame_num % PDF_EVERY == 0:
            save_fig(pdf_dir, name, frame_num, river_masks[i], agent_r, agent_c, "pdf")

        if i % 2000 == 0:
            print(f"    frame {i}/{N_FRAMES} ({time.time()-t0:.1f}s)")

    net_dx = (agent_c.astype(float) - init_c.astype(float)) * CELL_SIZE
    pct_down = np.sum(net_dx > 0) / N_AGENTS * 100
    print(f"  [{name}] Agents done in {time.time()-t0:.1f}s")
    print(f"  mean dx = {net_dx.mean():+.0f} m, {pct_down:.1f}% downstream\n")

    drift_results[name] = net_dx.copy()

# ── Save drift data as npz ──────────────────────────────────────────────
# Load any previously saved drift data so we can merge
npz_path = os.path.join(BASE, "drift_data.npz")
if os.path.exists(npz_path):
    prev = dict(np.load(npz_path))
    for key in ["symmetric_dx", "asymmetric_dx", "upstream_dx"]:
        sname = key.replace("_dx", "").capitalize()
        if sname not in drift_results and key in prev:
            drift_results[sname] = prev[key]

save_dict = {
    "cell_size": CELL_SIZE, "N_AGENTS": N_AGENTS,
    "W": W, "alpha_raw": alpha * W, "kl_myr": 12.0,
    "cutoff_dist_W": cutoff_dist / W,
    "sim_steps": SIM_STEPS, "warmup_steps": WARMUP,
}
for sname in ["Symmetric", "Asymmetric", "Upstream"]:
    if sname in drift_results:
        save_dict[f"{sname.lower()}_dx"] = drift_results[sname]
np.savez_compressed(npz_path, **save_dict)
print(f"Drift data saved: {npz_path}")

# ── KDE plot ────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(5, 3.5))

kde_configs = [
    ("Symmetric", "#666666"),
    ("Asymmetric", "#fb3333"),
    ("Upstream", "#3365ff"),
]
for sname, color in kde_configs:
    if sname not in drift_results:
        continue
    dx_km = drift_results[sname] / 1000.0
    kde = gaussian_kde(dx_km, bw_method=0.3)
    x_eval = np.linspace(dx_km.min() - 1, dx_km.max() + 1, 500)
    ax.plot(x_eval, kde(x_eval), color=color, linewidth=2, label=sname)
    ax.fill_between(x_eval, kde(x_eval), alpha=0.15, color=color)
    ax.axvline(dx_km.mean(), color=color, linewidth=1, linestyle='--', alpha=0.7)

ax.axvline(0, color='gray', linewidth=0.5, linestyle='--')
ax.set_xlabel("Along-valley migration (km)")
ax.set_ylabel("Density")
ax.legend()
ax.set_xlim(-12, 12)
fig.tight_layout()
fig.savefig(os.path.join(BASE, "migration_kde.pdf"), dpi=300, bbox_inches='tight')
plt.close(fig)
print(f"KDE plot saved: {BASE}/migration_kde.pdf")

# ── Combined mp4 from PNGs ──────────────────────────────────────────────
print("Making combined mp4...")
combined_dir = os.path.join(BASE, "combined_png")
os.makedirs(combined_dir, exist_ok=True)

dirs = {s: os.path.join(BASE, s.lower(), "png") for s in ["Symmetric", "Asymmetric", "Upstream"]}
# Use shortest frame list
all_frame_lists = []
for d in dirs.values():
    if os.path.isdir(d):
        all_frame_lists.append(sorted(f for f in os.listdir(d) if f.endswith(".png")))
frames = min(all_frame_lists, key=len) if all_frame_lists else []

for fname in frames:
    imgs = []
    for sname in ["Symmetric", "Asymmetric", "Upstream"]:
        p = os.path.join(dirs[sname], fname)
        if os.path.exists(p):
            imgs.append(Image.open(p))
    if len(imgs) == 3:
        w, h = imgs[0].size
        combined = Image.new("RGB", (w, h * 3))
        for j, im in enumerate(imgs):
            combined.paste(im, (0, h * j))
        combined.save(os.path.join(combined_dir, fname))

mp4_path = os.path.join(BASE, "all_scenarios.mp4")
subprocess.run([
    "ffmpeg", "-y", "-framerate", "17",
    "-pattern_type", "glob",
    "-i", os.path.join(combined_dir, "frame_*.png"),
    "-c:v", "libx264", "-preset", "slow", "-crf", "22",
    "-pix_fmt", "yuv420p",
    "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
    mp4_path
], check=True, capture_output=True)
sz = os.path.getsize(mp4_path) / 1e6
print(f"mp4 saved: {mp4_path} ({sz:.1f} MB)")

print("\n=== All outputs in", BASE, "===")
print("  symmetric/png/    - PNG every 100 frames")
print("  asymmetric/png/   - PNG every 100 frames")
print("  upstream/png/     - PNG every 100 frames")
print("  */pdf/            - PDF every 1000 frames")
print("  drift_data.npz    - migration distributions")
print("  migration_kde.pdf - KDE comparison plot")
print("  all_scenarios.mp4 - combined video")
