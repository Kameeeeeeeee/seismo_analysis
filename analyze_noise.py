import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seismo_utils import read_measurement_csv, detrend_mean, rms, compute_fft_mag, ensure_dir, save_fig

def analyze_noise(csv_path: str, out_dir: str):
    sig = read_measurement_csv(csv_path)
    ensure_dir(out_dir)

    x = detrend_mean(sig.x)
    r = rms(x)

    # Time plot
    plt.figure()
    plt.plot(sig.t, x)
    plt.xlabel("time (s)")
    plt.ylabel("az (m/s^2), detrended")
    plt.title(f"Noise - time series - {sig.name} (Fs={sig.fs:.1f} Hz)")
    save_fig(os.path.join(out_dir, f"{sig.name}_noise_time.png"))

    # FFT magnitude
    f, mag = compute_fft_mag(sig.t, x, sig.fs)
    plt.figure()
    plt.semilogy(f[1:], mag[1:])  # skip DC
    plt.xlabel("frequency (Hz)")
    plt.ylabel("FFT magnitude (approx)")
    plt.title(f"Noise - spectrum - {sig.name}")
    plt.xlim(0, min(100.0, 0.5 * sig.fs))
    save_fig(os.path.join(out_dir, f"{sig.name}_noise_spectrum.png"))

    # Save results
    res = {
        "file": os.path.basename(csv_path),
        "Fs_hz": sig.fs,
        "duration_s": float(sig.t[-1] - sig.t[0]),
        "RMS_ms2": r,
    }
    pd.DataFrame([res]).to_csv(os.path.join(out_dir, f"{sig.name}_noise_results.csv"), index=False)
    return res

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to noise CSV")
    ap.add_argument("--out", default="out", help="Output directory")
    args = ap.parse_args()
    analyze_noise(args.csv, args.out)

# python analyze_noise.py data/noise_1.csv --out out/noise
