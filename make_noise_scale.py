import os, glob, re
import numpy as np
import pandas as pd
from seismo_utils import read_measurement_csv, detrend_mean, rms, ensure_dir

def scenario_from_name(name: str) -> str:
    # expects noise_S2_steps_rep1.csv or similar
    # returns noise_S2_steps
    name = os.path.splitext(os.path.basename(name))[0]
    name = re.sub(r"_rep\d+$", "", name)
    return name

def main(data_dir="data", out_dir="out/noise_scale"):
    ensure_dir(out_dir)
    rows = []
    for path in sorted(glob.glob(os.path.join(data_dir, "noise*.csv"))):
        sig = read_measurement_csv(path)

        # ignore first 2 seconds
        mask = sig.t >= (sig.t[0] + 2.0)
        x = detrend_mean(sig.x[mask])

        rows.append({
            "file": os.path.basename(path),
            "scenario": scenario_from_name(path),
            "Fs_hz": sig.fs,
            "duration_s_used": float(sig.t[mask][-1] - sig.t[mask][0]),
            "RMS_ms2": rms(x),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(out_dir, "noise_rms_all.csv"), index=False)

    # aggregate
    agg = df.groupby("scenario")["RMS_ms2"].agg(["mean", "std", "count"]).reset_index()

    # baseline = first scenario that contains 'S1' or 'quiet'
    base = None
    for s in agg["scenario"]:
        if "S1" in s or "quiet" in s:
            base = float(agg.loc[agg["scenario"] == s, "mean"].iloc[0])
            break
    if base is None and len(agg) > 0:
        base = float(agg["mean"].iloc[0])

    agg["relative_to_baseline"] = agg["mean"] / base if base and base > 0 else np.nan
    agg = agg.sort_values("mean")

    agg.to_csv(os.path.join(out_dir, "noise_scale_summary.csv"), index=False)
    print(agg)

if __name__ == "__main__":
    main()
