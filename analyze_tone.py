import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seismo_utils import (
    read_measurement_csv, detrend_mean, rms, compute_fft_mag,
    parse_freq_from_filename, ensure_dir, save_fig
)

def amplitude_at_freq_fft(f: np.ndarray, mag: np.ndarray, target_hz: float):
    idx = int(np.argmin(np.abs(f - target_hz)))
    return float(f[idx]), float(mag[idx])

def analyze_tone(csv_path: str, out_dir: str, tone_hz: float | None = None):
    sig = read_measurement_csv(csv_path)
    ensure_dir(out_dir)

    x = detrend_mean(sig.x)
    r = rms(x)

    if tone_hz is None:
        tone_hz = parse_freq_from_filename(sig.name)

    f, mag = compute_fft_mag(sig.t, x, sig.fs)

    amp_f = None
    amp_val = None
    if tone_hz is not None:
        amp_f, amp_val = amplitude_at_freq_fft(f, mag, tone_hz)

    # Time plot
    plt.figure()
    plt.plot(sig.t, x)
    plt.xlabel("time (s)")
    plt.ylabel("az (m/s^2), detrended")
    title = f"Tone - time series - {sig.name} (Fs={sig.fs:.1f} Hz)"
    if tone_hz is not None:
        title += f"\nExpected tone ~ {tone_hz:.2f} Hz"
    plt.title(title)
    save_fig(os.path.join(out_dir, f"{sig.name}_tone_time.png"))

    # Spectrum
    plt.figure()
    plt.semilogy(f[1:], mag[1:])
    plt.xlabel("frequency (Hz)")
    plt.ylabel("FFT magnitude (approx)")
    t2 = f"Tone - spectrum - {sig.name}"
    if tone_hz is not None and amp_f is not None:
        t2 += f"\nAmp@{amp_f:.2f}Hz = {amp_val:.4g}"
        plt.axvline(amp_f)
    plt.title(t2)
    plt.xlim(0, min(100.0, 0.5 * sig.fs))
    save_fig(os.path.join(out_dir, f"{sig.name}_tone_spectrum.png"))

    res = {
        "file": os.path.basename(csv_path),
        "Fs_hz": sig.fs,
        "duration_s": float(sig.t[-1] - sig.t[0]),
        "RMS_ms2": r,
        "tone_hz": float(tone_hz) if tone_hz is not None else np.nan,
        "fft_amp_at_tone": float(amp_val) if amp_val is not None else np.nan,
    }
    pd.DataFrame([res]).to_csv(os.path.join(out_dir, f"{sig.name}_tone_results.csv"), index=False)
    return res

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to tone CSV (filename should contain like 12Hz)")
    ap.add_argument("--out", default="out", help="Output directory")
    ap.add_argument("--tone_hz", type=float, default=None, help="Override tone frequency if not in filename")
    args = ap.parse_args()
    analyze_tone(args.csv, args.out, args.tone_hz)
