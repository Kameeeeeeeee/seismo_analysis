import os
import pandas as pd

def convert_mobile_csv_to_seismo_format(
    in_csv: str,
    out_csv: str,
    time_col: str = "Time (s)",
    z_col: str = "Linear Acceleration z (m/s^2)",
    drop_initial_zeros: bool = True,
):
    # Важно: engine="python" помогает при странных кавычках/разделителях
    df = pd.read_csv(in_csv, engine="python")

    if time_col not in df.columns:
        raise ValueError(f"Не найден столбец времени '{time_col}'. Есть: {list(df.columns)}")
    if z_col not in df.columns:
        raise ValueError(f"Не найден столбец Z '{z_col}'. Есть: {list(df.columns)}")

    t = pd.to_numeric(df[time_col], errors="coerce")
    az = pd.to_numeric(df[z_col], errors="coerce")

    out = pd.DataFrame({"time_s": t, "az_ms2": az}).dropna().reset_index(drop=True)

    # Часто первые строки бывают нули, их можно убрать
    if drop_initial_zeros:
        # убираем ведущий участок, где |az| очень маленький
        # (это не обязательно, но помогает для impulse)
        eps = 1e-6
        first_idx = out.index[(out["az_ms2"].abs() > eps)].min()
        if pd.notna(first_idx) and first_idx > 0:
            out = out.loc[first_idx:].reset_index(drop=True)

    # Нормализуем время, чтобы старт был с 0
    out["time_s"] = out["time_s"] - float(out["time_s"].iloc[0])

    # Добавляем sample
    out.insert(0, "sample", range(len(out)))

    # Сохраняем в нужном формате
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    out.to_csv(out_csv, index=False, float_format="%.6f")

    print(f"Готово: {out_csv}")
    print(f"Строк: {len(out)}, длительность: {out['time_s'].iloc[-1]:.3f} s")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("in_csv", help="Входной CSV из мобильного приложения")
    ap.add_argument("out_csv", help="Выходной CSV в формате sample,time_s,az_ms2")
    ap.add_argument("--time_col", default="Time (s)")
    ap.add_argument("--z_col", default="Linear Acceleration z (m/s^2)")
    ap.add_argument("--keep_initial_zeros", action="store_true")
    args = ap.parse_args()

    convert_mobile_csv_to_seismo_format(
        args.in_csv,
        args.out_csv,
        time_col=args.time_col,
        z_col=args.z_col,
        drop_initial_zeros=(not args.keep_initial_zeros),
    )

# python mobile_to_seismo.py "data/phyphox/fb_20sm/Raw Data.csv" "data/football_20sm_phyphox.csv"
# python mobile_to_seismo.py "data/phyphox/fb_40sm/Raw Data.csv" "data/football_40sm_phyphox.csv"
# python mobile_to_seismo.py "data/phyphox/ten_20sm/Raw Data.csv" "data/tennis_20sm_phyphox.csv"
# python mobile_to_seismo.py "data/phyphox/ten_40sm/Raw Data.csv" "data/tennis_40sm_phyphox.csv"
