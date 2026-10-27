"""Every C1 metric, in one place. Implements criteria/sealed/c1-glimmer-frequency-response-v4.yaml.

Pure numpy, no model code, so every function can be tested on its own.
This file is protected once committed: propose changes, never edit inline.

Direction words. Short N means high frequency (1/N cycles per token). A curve whose gain
falls as N grows passes high frequencies: high-pass (HP). A curve whose gain rises as N
grows passes low frequencies: low-pass (LP).
"""
import numpy as np

WINDOW_START = 8192
WINDOW_END = 32768
NOISE_PER_SIDE = 8
MIN_BELOW = 3                       # N = 4096 has k0 = 6, so only bins 2, 3, 4 lie below it
PASSBAND = 0.7                      # a cutoff plateau must reach this fraction of the curve maximum
PLATEAU_SPREAD = 1.15               # the three plateau points agree within 15 percent
SHARPNESS = 4.0                     # the curve goes from 0.9 to 0.5 of the plateau within a factor of 4 in N
FLAT_MIN = 0.67                     # flat: every valid normalized point at least 0.67 of the maximum
HEAD_DIM = 128

# ---------- pulse trains ----------

def periodic_positions(N: int, T: int, seed: int = 0) -> np.ndarray:
    """Every N tokens from an offset in [0, N) drawn by a fresh numpy default_rng(seed) for each N."""
    offset = int(np.random.default_rng(seed).integers(0, N))
    return np.arange(offset, T, N, dtype=np.int64)


def aperiodic_positions(count: int, T: int, seed: int = 1) -> np.ndarray:
    """The same number of pulses at random distinct positions in [0, T), fresh default_rng(seed)."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(T, size=count, replace=False)).astype(np.int64)


# ---------- analysis window and operator ----------

def noise_bins(N: int, W: int, per_side: int = NOISE_PER_SIDE):
    """Noise bins on both sides of the target bin k0 = W/N. Returns (bins, n_below).

    bins lists up to `per_side` integer bins below k0 (nearest first), then `per_side` above.
    At N = 2 the target is the Nyquist bin, so 2 * per_side bins below are used instead.
    Bins within 1 of any multiple of k0 (DC included) are skipped, so Hann leakage from any
    harmonic never reaches a noise bin. Both sides are kept because a smooth non-periodic
    drift leaks mostly into low bins: a floor taken only above k0 would miss it at long N.
    """
    if W % N:
        raise ValueError(f"window {W} does not hold a whole number of periods of N={N}")
    k0 = W // N
    if k0 < 5:
        raise ValueError(f"window holds only {k0} periods of N={N}; at least 5 are needed for noise bins")
    ks = np.arange(1, W // 2 + 1, dtype=np.int64)
    r = ks % k0
    candidates = ks[~(np.minimum(r, k0 - r) <= 1)]
    if 2 * k0 == W:                                     # N = 2: the target is the Nyquist bin, nothing lies above
        below, above = candidates[candidates < k0][::-1][:2 * per_side], candidates[:0]
    else:
        below = candidates[candidates < k0][::-1][:per_side]
        above = candidates[candidates > k0][:per_side]
    if len(below) < MIN_BELOW or (len(above) < per_side and 2 * k0 != W):
        raise ValueError(f"N={N}: {len(below)} noise bins below and {len(above)} above k0")
    return np.concatenate([below, above]).astype(np.float64), int(len(below))


def two_sided_floor(values: np.ndarray, n_below: int) -> np.ndarray:
    """Noise floor along axis 0: the larger of the median below k0 and the median above k0.
    At N = 2 every noise bin lies below k0 and the floor is their median."""
    if n_below == len(values):
        return np.median(values, axis=0)
    return np.maximum(np.median(values[:n_below], axis=0), np.median(values[n_below:], axis=0))


def analysis_frequencies(N: int, W: int):
    """Index 0 is the target frequency 1/N; then the noise frequencies k/W, below k0 first. Returns (freqs, n_below)."""
    bins, n_below = noise_bins(N, W)
    return np.concatenate([[1.0 / N], bins / W]), n_below


def hann(W: int) -> np.ndarray:
    """Periodic Hann window: an integer-bin sinusoid leaks only into its two neighbor bins."""
    n = np.arange(W, dtype=np.float64)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * n / W)


def analysis_basis(freqs: np.ndarray, t1: int, W: int):
    """Effective cos and sin bases, shape [W, F], for: remove a least-squares line, apply Hann, take the Fourier sum.

    Component(x) = sum_t b_t * hann_t * (P x)_t, where P removes the best-fit line a + b t.
    P is a symmetric projection, so Component(x) = (P (hann * b))^T x. The same operator is
    applied to the response and to the input pulse train, so a system that passes the
    input unchanged has gain exactly 1.
    """
    t = np.arange(t1, t1 + W, dtype=np.float64)
    q, _ = np.linalg.qr(np.stack([np.ones(W), t - t.mean()], axis=1))
    w = hann(W)[:, None]
    phase = 2.0 * np.pi * freqs[None, :] * t[:, None]
    out = []
    for b in (np.cos(phase), np.sin(phase)):
        wb = w * b
        out.append(wb - q @ (q.T @ wb))
    return out[0], out[1]


def component(x: np.ndarray, cos_b: np.ndarray, sin_b: np.ndarray) -> np.ndarray:
    """|Component| at each frequency for a 1-D signal x over the window. Shape [F]."""
    re = cos_b.T @ x
    im = -(sin_b.T @ x)
    return np.sqrt(re * re + im * im)


def train_component(positions: np.ndarray, amplitude: float, cos_b: np.ndarray, sin_b: np.ndarray,
                    t1: int, T: int) -> float:
    """Target-frequency component of the input pulse train, through the same operator as the response."""
    x = np.zeros(T - t1, dtype=np.float64)
    p = positions[(positions >= t1) & (positions < T)] - t1
    x[p] = amplitude
    return float(component(x, cos_b[:, :1], sin_b[:, :1])[0])


def head_magnitudes(response: np.ndarray, cos_b: np.ndarray, sin_b: np.ndarray, n_heads: int) -> np.ndarray:
    """Reference implementation. response [W, n_heads*d] -> |Fourier component| per frequency and head, [F, n_heads]."""
    re = cos_b.T @ response
    im = -(sin_b.T @ response)
    power = re ** 2 + im ** 2
    return np.sqrt(power.reshape(power.shape[0], n_heads, -1).sum(axis=2))


# ---------- per-book quantities ----------

def median_over_mean_power(dof: int) -> float:
    """Median over mean of a chi-square variable with `dof` degrees of freedom (Wilson-Hilferty)."""
    return (1.0 - 2.0 / (9.0 * dof)) ** 3


def gains(target: np.ndarray, floor: np.ndarray, input_mag: float, head_dim: int = HEAD_DIM):
    """target, floor: [layers, heads] magnitudes. Returns gain and noise-corrected gain, [layers, heads].

    floor is the two-sided noise-bin magnitude floor per head. A head's noise power is chi-square with
    2 * head_dim degrees of freedom, so mean noise power = floor^2 / median_over_mean_power.
    """
    gain = target / input_mag
    noise_power = floor ** 2 / median_over_mean_power(2 * head_dim)
    corrected = np.sqrt(np.maximum(target ** 2 - noise_power, 0.0)) / input_mag
    return gain, corrected


def aggregate_snr(target: np.ndarray, bin_power: np.ndarray, n_below: int) -> float:
    """Whole-model SNR: sqrt(summed target power over all heads / two-sided floor of summed bin power).

    target [layers, heads] magnitudes; bin_power [bins] = summed power over all heads at each noise bin.
    """
    noise = float(two_sided_floor(np.asarray(bin_power, dtype=np.float64), n_below))
    signal = float(np.sum(np.asarray(target, dtype=np.float64) ** 2))
    return float(np.sqrt(signal / noise)) if noise > 0 else float("inf")


def group_means(gain: np.ndarray, layer_ids, global_ids) -> dict:
    """Mean gain over all heads, local-layer heads, and global-layer heads."""
    g = set(global_ids)
    g_mask = np.array([l in g for l in layer_ids])
    return {"whole": float(gain.mean()), "local": float(gain[~g_mask].mean()), "global": float(gain[g_mask].mean())}


# ---------- set curves and outcomes ----------

def spearman(x, y) -> float:
    def rank(a):
        a = np.asarray(a, dtype=np.float64)
        order = a.argsort(kind="stable")
        r = np.empty(len(a))
        r[order] = np.arange(len(a))
        for v in np.unique(a):
            idx = a == v
            r[idx] = r[idx].mean()
        return r
    rx, ry = rank(x), rank(y)
    rx, ry = rx - rx.mean(), ry - ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


def normalize(curve: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Divide by the maximum over valid points; invalid points become NaN."""
    if not valid.any():
        return np.full(len(curve), np.nan)
    peak = np.max(curve[valid])
    return np.where(valid, curve / peak, np.nan)


def _crossing(grid, c, a, b, level):
    la, lb = np.log(grid[a]), np.log(grid[b])
    return float(np.exp(la + (c[a] - level) / (c[a] - c[b]) * (lb - la)))


def cutoff(grid, norm_curve, valid) -> dict:
    """Symmetric cutoff test. Returns direction ("HP", "LP" or None), cutoff_N, and the evidence.

    HP: plateau = the three shortest valid N. N_c is where the curve, scanning toward longer N,
        crosses below half the plateau mean and stays below at every longer valid point.
    LP: plateau = the three longest valid N. N_c is where the curve, scanning toward shorter N,
        crosses below half the plateau mean and stays below at every shorter valid point.
    A direction counts only if its plateau is a real passband:
      - the plateau mean is at least PASSBAND (0.7) of the curve maximum;
      - the three plateau points agree within PLATEAU_SPREAD (max / min at most 1.15);
      - the transition is sharp: N_90, the crossing of 0.9 x plateau nearest N_c on the plateau
        side, and N_c differ by at most SHARPNESS (4x).
    So a power law, which has no plateau and no sharp knee, has no cutoff and falls through to T.
    Crossings are interpolated linearly in log N between consecutive valid points. The two
    directions cannot both hold. Needs at least 6 valid points.
    """
    grid = np.asarray(grid, dtype=np.float64)
    c = np.asarray(norm_curve, dtype=np.float64)
    ok = np.where(valid)[0]
    none = {"direction": None, "cutoff_N": None, "N_90": None}
    if len(ok) < 6:
        return none
    peak = float(np.max(c[ok]))

    def plateau_ok(idx):
        p = c[idx]
        return float(np.mean(p)) >= PASSBAND * peak and float(np.max(p) / np.min(p)) <= PLATEAU_SPREAD

    if plateau_ok(ok[:3]):
        p = float(np.mean(c[ok[:3]]))
        for j in range(len(ok) - 1):
            a, b = ok[j], ok[j + 1]
            if c[a] >= 0.5 * p > c[b] and np.all(c[ok[j + 1:]] < 0.5 * p):
                nc = _crossing(grid, c, a, b, 0.5 * p)
                i = max(i for i in range(j + 1) if c[ok[i]] >= 0.9 * p)
                n90 = _crossing(grid, c, ok[i], ok[i + 1], 0.9 * p) if c[ok[i + 1]] < 0.9 * p else float(grid[ok[i + 1]])
                if nc / n90 <= SHARPNESS:
                    return {"direction": "HP", "cutoff_N": nc, "N_90": n90}
                return dict(none, N_90=n90, rejected="HP transition not sharp")
    if plateau_ok(ok[-3:]):
        p = float(np.mean(c[ok[-3:]]))
        for j in range(len(ok) - 1, 0, -1):
            a, b = ok[j], ok[j - 1]
            if c[a] >= 0.5 * p > c[b] and np.all(c[ok[:j]] < 0.5 * p):
                nc = _crossing(grid, c, b, a, 0.5 * p)
                i = min(i for i in range(j, len(ok)) if c[ok[i]] >= 0.9 * p)
                n90 = _crossing(grid, c, ok[i - 1], ok[i], 0.9 * p) if c[ok[i - 1]] < 0.9 * p else float(grid[ok[i - 1]])
                if n90 / nc <= SHARPNESS:
                    return {"direction": "LP", "cutoff_N": nc, "N_90": n90}
                return dict(none, N_90=n90, rejected="LP transition not sharp")
    return none


def resonance(grid, norm_curve, valid) -> dict:
    """Peak tests for one set curve. The peak must be interior on the full grid, stand 1.5x above the
    median of valid points, and have at least 2 valid points on each side at or below peak / 1.5."""
    n = len(grid)
    c = np.asarray(norm_curve, dtype=np.float64)
    vi = np.where(valid)[0]
    peak_i = int(vi[np.argmax(c[vi])])
    peak = float(c[peak_i])
    left, right = vi[vi < peak_i], vi[vi > peak_i]
    low_left = int(np.sum(c[left] <= peak / 1.5))
    low_right = int(np.sum(c[right] <= peak / 1.5))
    sides_ok = low_left >= 2 and low_right >= 2
    return {"peak_index": peak_i, "peak_N": grid[peak_i],
            "peak_over_median": float(peak / np.median(c[vi])),
            "interior": bool(2 <= peak_i <= n - 3),
            "valid_left": int(len(left)), "valid_right": int(len(right)),
            "low_left": low_left, "low_right": low_right, "sides_fall": bool(sides_ok)}


def classify_c1(grid, set_curves: dict, set_valid: dict) -> dict:
    """set_curves: {set: normalized whole-model curve}. Outcome R, HP, LP, T, F, S or U with the evidence.

    U  inconclusive: any set has fewer than half of the grid points valid. Not a kill.
    R  resonance: on all sets, an interior peak passing every test in `resonance`, peak indices within 1.
    HP high-pass / LP low-pass: not R; on all sets the cutoff direction is the same and the three
       N_c agree within 1.5x.
    T  tilt: none of the above; Spearman rho of gain vs N has |rho| >= 0.8 with the same sign on all
       sets. rho < 0 is a high-pass tilt, rho > 0 a low-pass tilt.
    F  flat: none of the above, and on every set every valid normalized point is at least FLAT_MIN.
    S  structured: none of the above, not flat, and every pair of set curves has Spearman rho >= 0.8
       over shared valid points (a replicated shape such as a notch, described in C1b).
    U  inconclusive also covers: none of the above and the shape does not replicate across sets.
    """
    grid = list(grid)
    n = len(grid)
    sets = sorted(set_curves)
    per = {}
    for s in sets:
        c, v = set_curves[s], set_valid[s]
        info = {"valid_points": int(v.sum()), "invalid_N": [grid[i] for i in range(n) if not v[i]]}
        if v.sum() >= 1:
            info.update(resonance(grid, c, v))
            info.update(cutoff(grid, c, v))
            vi = np.where(v)[0]
            info["rho_gain_vs_N"] = spearman(np.array(grid)[vi], c[vi])
            info["min_normalized"] = float(np.min(c[vi]))
        per[s] = info
    rep = {}
    for i, a in enumerate(sets):
        for b in sets[i + 1:]:
            both = set_valid[a] & set_valid[b]
            rho = spearman(set_curves[a][both], set_curves[b][both]) if both.sum() > 2 else None
            rep[f"{a}-{b}"] = {"rho": rho, "at_least_0.8": bool(rho is not None and rho >= 0.8)}
    enough = all(per[s]["valid_points"] >= n / 2 for s in sets)
    outcome, direction, reason = "U", None, "fewer than half of the grid points valid in some set"
    if enough:
        dirs = {per[s]["direction"] for s in sets}
        cuts = [per[s]["cutoff_N"] for s in sets]
        rhos = [per[s]["rho_gain_vs_N"] for s in sets]
        if (all(per[s]["interior"] and per[s]["peak_over_median"] >= 1.5 and per[s]["sides_fall"] for s in sets)
                and max(per[s]["peak_index"] for s in sets) - min(per[s]["peak_index"] for s in sets) <= 1):
            outcome, reason = "R", None
        elif len(dirs) == 1 and None not in dirs and max(cuts) / min(cuts) <= 1.5:
            outcome, reason = dirs.pop(), None
        elif all(r <= -0.8 for r in rhos):
            outcome, direction, reason = "T", "HP", None
        elif all(r >= 0.8 for r in rhos):
            outcome, direction, reason = "T", "LP", None
        elif all(per[s]["min_normalized"] >= FLAT_MIN for s in sets):
            outcome, reason = "F", None
        elif all(r["at_least_0.8"] for r in rep.values()):
            outcome, reason = "S", None
        else:
            reason = "not flat, and the shape does not replicate across sets"
    return {"outcome": outcome, "tilt_direction": direction, "inconclusive_reason": reason,
            "enough_valid": bool(enough), "per_set": per, "shape_replication": rep}


def notch_index(grid, curve, at=2048, neighbors=(1536, 3072)):
    """Exploratory, not a gate: gain at `at` divided by the mean of its two neighbors."""
    g = list(grid)
    if at not in g or any(x not in g for x in neighbors):
        return None
    mid = curve[g.index(at)]
    side = np.mean([curve[g.index(n)] for n in neighbors])
    return float(mid / side) if side > 0 else float("nan")


def box_filter_null(grid, L: int = 2048) -> list:
    """Reference only: |H| of an L-token moving average at f = 1/N. Zero wherever N divides L."""
    out = []
    for N in grid:
        f = 1.0 / N
        x = f * L
        out.append(0.0 if abs(x - round(x)) < 1e-12 else float(abs(np.sin(np.pi * x) / (L * np.sin(np.pi * f)))))
    return out
