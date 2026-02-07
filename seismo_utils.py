import os
import re
from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

@dataclass
class Signal:
    t: np.ndarray
    x: np.ndarray
    fs: float
    name: str

def read_measurement_csv(path: str) -> Signal:
    df = pd.read_csv(path)
    if not {"time_s", "az_ms2"}.issubset(df.columns):
        raise ValueError(f"CSV must contain columns time_s and az_ms2, got {list(df.columns)}")

    t = df["time_s"].to_numpy(dtype=float)
    x = df["az_ms2"].to_numpy(dtype=float)

    if len(t) < 10:
        raise ValueError("Too few samples")

    dt = np.diff(t)
    dt_med = float(np.median(dt))
    if dt_med <= 0:
        raise ValueError("Non-positive dt detected")

    fs = 1.0 / dt_med
    name = os.path.splitext(os.path.basename(path))[0]
    return Signal(t=t, x=x, fs=fs, name=name)

def detrend_mean(x: np.ndarray) -> np.ndarray:
    return x - float(np.mean(x))

def moving_average(x: np.ndarray, win: int) -> np.ndarray:
    win = max(1, int(win))
    if win == 1:
        return x.copy()
    kernel = np.ones(win, dtype=float) / win
    return np.convolve(x, kernel, mode="same")

def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x))))

def hann(n: int) -> np.ndarray:
    if n <= 1:
        return np.ones(n, dtype=float)
    return 0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / (n - 1))

def compute_fft_mag(t: np.ndarray, x: np.ndarray, fs: float):
    # Windowed FFT magnitude spectrum
    n = len(x)
    w = hann(n)
    xw = x * w
    X = np.fft.rfft(xw)
    f = np.fft.rfftfreq(n, d=1.0/fs)
    mag = np.abs(X) / (np.sum(w) / 2.0)  # approximate amplitude correction
    return f, mag

def peak_and_bw_3db(f: np.ndarray, mag: np.ndarray, fmin=0.5, fmax=80.0):
    band = (f >= fmin) & (f <= fmax)
    fb = f[band]
    mb = mag[band]
    if len(fb) < 10:
        return None

    i0 = int(np.argmax(mb))
    f_peak = float(fb[i0])
    a_peak = float(mb[i0])
    if a_peak <= 0:
        return None

    thr = a_peak / np.sqrt(2.0)

    # find left crossing
    il = i0
    while il > 0 and mb[il] >= thr:
        il -= 1
    f_left = float(fb[il]) if il < i0 else float(fb[0])

    # find right crossing
    ir = i0
    while ir < len(mb) - 1 and mb[ir] >= thr:
        ir += 1
    f_right = float(fb[ir]) if ir > i0 else float(fb[-1])

    bw = float(f_right - f_left)
    return {"f_peak": f_peak, "a_peak": a_peak, "f_left": f_left, "f_right": f_right, "bw": bw}

def simple_peak_indices(x: np.ndarray, min_distance: int = 10, min_prom: float = 0.0):
    # local maxima with a crude prominence filter
    n = len(x)
    if n < 3:
        return np.array([], dtype=int)

    peaks = []
    last = -10**9
    for i in range(1, n-1):
        if x[i] > x[i-1] and x[i] >= x[i+1]:
            if i - last < min_distance:
                continue
            # crude prominence: higher than neighbors by min_prom
            if (x[i] - max(x[i-1], x[i+1])) >= min_prom:
                peaks.append(i)
                last = i
    return np.array(peaks, dtype=int)

def estimate_ringdown_params(t: np.ndarray, x: np.ndarray, fs: float):
    # Use peaks on absolute value (envelope-like) for decay and on raw for period
    x0 = detrend_mean(x)

    # Smooth for stable peak picking
    xs = moving_average(x0, win=max(3, int(fs * 0.01)))  # 10 ms window

    # Period via peaks on xs (raw)
    prom = 0.05 * np.std(xs)
    peaks = simple_peak_indices(xs, min_distance=max(3, int(fs * 0.02)), min_prom=prom)

    if len(peaks) < 6:
        return None

    tp = t[peaks]
    Ap = np.abs(xs[peaks])

    # Use middle portion to avoid initial impact transient
    k0 = 1
    k1 = min(len(tp) - 1, 12)
    if k1 - k0 < 4:
        k0 = 0
        k1 = len(tp) - 1

    # f0 from average period over several cycles
    # Take differences between consecutive peaks
    dT = np.diff(tp[k0:k1])
    T = float(np.median(dT))
    f0 = 1.0 / T if T > 0 else np.nan

    # Log decrement using peaks separated by n cycles
    n_cycles = min(10, (k1 - k0 - 1))
    if n_cycles < 2:
        return None

    A1 = float(Ap[k0])
    A2 = float(Ap[k0 + n_cycles])
    if A1 <= 0 or A2 <= 0 or A2 >= A1:
        # try another pair later
        idx1 = k0
        idx2 = k0 + n_cycles
        found = False
        for shift in range(0, 5):
            if idx2 + shift < len(Ap):
                a1 = float(Ap[idx1 + shift])
                a2 = float(Ap[idx2 + shift])
                if a1 > 0 and a2 > 0 and a2 < a1:
                    A1, A2 = a1, a2
                    found = True
                    break
        if not found:
            return None

    delta = (1.0 / n_cycles) * float(np.log(A1 / A2))
    Q = float(np.pi / delta) if delta > 0 else np.nan

    return {
        "f0_hz": f0,
        "T_s": T,
        "log_decrement": delta,
        "Q": Q,
        "peaks_idx": peaks,
    }

def parse_freq_from_filename(name: str):
    # expects patterns like "12Hz" or "12.5Hz"
    m = re.search(r"(\d+(?:\.\d+)?)\s*Hz", name, flags=re.IGNORECASE)
    if not m:
        return None
    return float(m.group(1))

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_fig(path: str):
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
