import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seismo_utils import (
    read_measurement_csv, detrend_mean, compute_fft_mag, parse_freq_from_filename,
    ensure_dir, save_fig
)

def amp_at_target(sig, target_hz: float):
    x = detrend_mean(sig.x)
    f, mag = compute_fft_mag(sig.t, x, sig.fs)
    idx = int(np.argmin(np.abs(f - target_hz)))
    return float(f[idx]), float(mag[idx]), float(sig.fs)

def find_minus3db_band(freqs, amps):
    # -3 dB points in amplitude scale -> amp / sqrt(2)
    if len(freqs) < 5:
        return None
    i_max = int(np.argmax(amps))
    a_max = float(amps[i_max])
    if a_max <= 0:
        return None
    thr = a_max / np.sqrt(2.0)

    # left
    i = i_max
    left = None
    while i > 0:
        if amps[i] >= thr and amps[i-1] < thr:
            left = freqs[i-1]
            break
        i -= 1

    # right
    i = i_max
    right = None
    while i < len(amps)-1:
        if amps[i] >= thr and amps[i+1] < thr:
            right = freqs[i+1]
            break
        i += 1

    if left is None or right is None:
        return None
    return float(freqs[i_max]), a_max, float(left), float(right), float(right - left)

def analyze_sweep_folder(data_dir: str, out_dir: str):
    ensure_dir(out_dir)
    paths = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    rows = []

    for p in paths:
        name = os.path.splitext(os.path.basename(p))[0]
        tone_hz = parse_freq_from_filename(name)
        if tone_hz is None:
            continue
        sig = read_measurement_csv(p)
        f_bin, amp, fs = amp_at_target(sig, tone_hz)
        rows.append({
            "file": os.path.basename(p),
            "tone_hz": float(tone_hz),
            "fft_bin_hz": float(f_bin),
            "amp": float(amp),
            "Fs_hz": float(fs),
            "duration_s": float(sig.t[-1] - sig.t[0]),
        })

    if not rows:
        raise RuntimeError("No CSV files with 'Hz' in filename were found.")

    df = pd.DataFrame(rows).sort_values("tone_hz").reset_index(drop=True)
    df.to_csv(os.path.join(out_dir, "sweep_points.csv"), index=False)

    freqs = df["tone_hz"].to_numpy(dtype=float)
    amps = df["amp"].to_numpy(dtype=float)

    band = find_minus3db_band(freqs, amps)
    summary = {}
    if band is not None:
        f_res, a_max, f_l, f_r, df_bw = band
        Q = f_res / df_bw if df_bw > 0 else np.nan
        summary = {
            "f_res_hz": f_res,
            "A_max": a_max,
            "f_left_hz": f_l,
            "f_right_hz": f_r,
            "bw_hz": df_bw,
            "Q_bw": Q,
        }

    # Plot AChX
    plt.figure()
    plt.plot(freqs, amps, marker="o")
    plt.xlabel("frequency (Hz)")
    plt.ylabel("response amplitude (FFT mag at tone)")
    title = "Amplitude-frequency response (AChX)"
    if summary:
        title += f"\nres={summary['f_res_hz']:.2f}Hz, Q={summary['Q_bw']:.1f}"
        plt.axvline(summary["f_res_hz"])
    plt.title(title)
    save_fig(os.path.join(out_dir, "sweep_achx.png"))

    if summary:
        pd.DataFrame([summary]).to_csv(os.path.join(out_dir, "sweep_summary.csv"), index=False)

    return df, summary

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", help="Folder with tone_*.csv files (filenames contain like 12Hz)")
    ap.add_argument("--out", default="out", help="Output directory")
    args = ap.parse_args()
    analyze_sweep_folder(args.data_dir, args.out)
