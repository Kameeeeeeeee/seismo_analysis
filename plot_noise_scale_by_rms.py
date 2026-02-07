# plot_noise_scale_by_rms.py
# Строит график "Шкала шумности по RMS" по CSV-файлу, который делает make_noise_scale.py
# Вход: out/noise_scale/noise_scale_summary.csv
# Выход: out/noise_scale/noise_scale_by_rms.png

import os
import pandas as pd
import matplotlib.pyplot as plt


def main(
    summary_csv: str = "out/noise_scale/noise_scale_summary.csv",
    out_png: str = "out/noise_scale/noise_scale_by_rms.png",
):
    if not os.path.isfile(summary_csv):
        raise FileNotFoundError(f"Не найден файл: {summary_csv}")

    df = pd.read_csv(summary_csv)

    # Ожидаемые столбцы: scenario, mean (и опционально std)
    if "scenario" not in df.columns or "mean" not in df.columns:
        raise ValueError(
            f"В {summary_csv} должны быть столбцы 'scenario' и 'mean'. Сейчас: {list(df.columns)}"
        )

    # Сортируем по среднему RMS (от тихого к шумному)
    df = df.sort_values("mean", ascending=True).reset_index(drop=True)

    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    plt.figure(figsize=(9, 5))
    plt.bar(df["scenario"], df["mean"])
    plt.ylabel("RMS (м/с^2)")
    plt.title("Шкала шумности по RMS")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close()

    print(f"График сохранен: {out_png}")


if __name__ == "__main__":
    main()
