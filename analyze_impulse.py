import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from seismo_utils import (
    read_measurement_csv,
    detrend_mean,
    rms,
    compute_fft_mag,
    estimate_ringdown_params,
    peak_and_bw_3db,
    ensure_dir,
    save_fig,
)


def analyze_impulse(csv_path: str, out_dir: str):
    sig = read_measurement_csv(csv_path)
    ensure_dir(out_dir)

    x = detrend_mean(sig.x)
    r = rms(x)

    params = estimate_ringdown_params(sig.t, x, sig.fs)
    if params is None:
        raise RuntimeError("Could not estimate ringdown parameters. Try longer recording or clearer oscillations.")

    peaks = params["peaks_idx"]

    # Time plot with peaks
    plt.figure()
    plt.plot(sig.t, x, label="az detrended")
    plt.plot(sig.t[peaks], x[peaks], "o", label="peaks")
    plt.xlabel("time (s)")
    plt.ylabel("az (m/s^2)")
    plt.title(
        f"Impulse ringdown - {sig.name}\n"
        f"Fs={sig.fs:.1f} Hz, f0={params['f0_hz']:.2f} Hz, delta={params['log_decrement']:.4f}, Q={params['Q']:.1f}"
    )
    plt.legend()
    save_fig(os.path.join(out_dir, f"{sig.name}_impulse_time_peaks.png"))

    # Spectrum on ringdown tail to reduce contamination from the impact transient.
    impact_idx = int(np.argmax(np.abs(x)))
    t_impact = float(sig.t[impact_idx])
    tail_start = t_impact + 0.2
    tail_end = min(t_impact + 3.0, float(sig.t[-1]))
    tail_mask = (sig.t >= tail_start) & (sig.t <= tail_end)
    if int(np.count_nonzero(tail_mask)) < 64:
        tail_mask = np.ones_like(sig.t, dtype=bool)
        tail_start = float(sig.t[0])
        tail_end = float(sig.t[-1])

    xt = x[tail_mask]
    tt = sig.t[tail_mask]
    f, mag = compute_fft_mag(tt, xt, sig.fs)
    pb = peak_and_bw_3db(f, mag, fmin=0.5, fmax=min(40.0, 0.45 * sig.fs))

    plt.figure()
    plt.semilogy(f[1:], mag[1:])
    plt.xlabel("frequency (Hz)")
    plt.ylabel("FFT magnitude (approx)")
    if pb is not None:
        plt.axvline(pb["f_left"], color="tab:orange", linestyle="--", linewidth=1.2, label=f"fL={pb['f_left']:.2f} Hz")
        plt.axvline(pb["f_peak"], color="tab:red", linestyle="-", linewidth=1.4, label=f"f0={pb['f_peak']:.2f} Hz")
        plt.axvline(pb["f_right"], color="tab:orange", linestyle="--", linewidth=1.2, label=f"fR={pb['f_right']:.2f} Hz")
        plt.title(
            f"Impulse - spectrum - {sig.name}\n"
            f"f_peak={pb['f_peak']:.2f} Hz, bw={pb['bw']:.2f} Hz (fL={pb['f_left']:.2f}, fR={pb['f_right']:.2f})"
        )
        plt.legend()
    else:
        plt.title(f"Impulse - spectrum - {sig.name}")
    plt.xlim(0, min(40.0, 0.5 * sig.fs))
    save_fig(os.path.join(out_dir, f"{sig.name}_impulse_spectrum.png"))

    # Save results
    res = {
        "file": os.path.basename(csv_path),
        "Fs_hz": sig.fs,
        "duration_s": float(sig.t[-1] - sig.t[0]),
        "RMS_ms2": r,
        "f0_hz": float(params["f0_hz"]),
        "T_s": float(params["T_s"]),
        "log_decrement": float(params["log_decrement"]),
        "Q": float(params["Q"]),
        "spec_tail_start_s": tail_start,
        "spec_tail_end_s": tail_end,
    }
    if pb is not None:
        res["spec_f_peak_hz"] = pb["f_peak"]
        res["spec_bw_hz"] = pb["bw"]
        res["spec_f_left_hz"] = pb["f_left"]
        res["spec_f_right_hz"] = pb["f_right"]
    else:
        res["spec_f_peak_hz"] = np.nan
        res["spec_bw_hz"] = np.nan
        res["spec_f_left_hz"] = np.nan
        res["spec_f_right_hz"] = np.nan
    pd.DataFrame([res]).to_csv(os.path.join(out_dir, f"{sig.name}_impulse_results.csv"), index=False)
    return res


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="Path to impulse CSV")
    ap.add_argument("--out", default="out", help="Output directory")
    args = ap.parse_args()
    analyze_impulse(args.csv, args.out)
