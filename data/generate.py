"""Bikin dataset sintetis mesin CNC untuk P2 (predictive maintenance).

Pola kegagalan meniru mode kegagalan nyata mesin perkakas (mirip dataset AI4I 2020):
tool aus, panas tidak terbuang, daya di luar batas, dan beban berlebih pada tool aus.
Lalu data sengaja dikotori supaya mirip kondisi lapangan.

Jalankan:  python data/generate.py   ->  data/mesin_cnc.csv
"""
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_BERSIH = 10_000      # jumlah baris unik (setelah drop duplikat)
N_DUPLIKAT = 150       # baris kembar yang sengaja diselipkan
OUT = Path(__file__).with_name("mesin_cnc.csv")

# batas strain (keausan x torsi) per tipe produk, tool aus + beban besar = rawan patah
K_RISIKO, B_RISIKO = 1.1, -6.5  # kemiringan dan intercept peluang gagal
BATAS_STRAIN = {"L": 11_000, "M": 12_000, "H": 13_000}


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def bersih(rng: np.random.Generator, n: int) -> pd.DataFrame:
    tipe = rng.choice(["L", "M", "H"], size=n, p=[0.6, 0.3, 0.1])
    suhu_udara = rng.normal(300.0, 2.0, n)
    suhu_proses = suhu_udara + 10.0 + rng.normal(0, 1.0, n)
    rpm = np.clip(rng.lognormal(np.log(1500), 0.12, n), 1150, 2900)
    # torsi turun kalau rpm naik (daya relatif konstan), plus variasi
    torsi = np.clip(40 + (1500 - rpm) * 0.045 + rng.normal(0, 10, n), 3.5, 77)
    keausan = rng.integers(0, 254, n)
    jam_servis = rng.integers(0, 721, n)
    shift = rng.choice(["pagi", "siang", "malam"], size=n, p=[0.4, 0.35, 0.25])

    daya = torsi * 2 * np.pi * rpm / 60
    delta = suhu_proses - suhu_udara
    strain = keausan * torsi
    batas = np.vectorize(BATAS_STRAIN.get)(tipe)

    # skor risiko kontinu dari tiap mode kegagalan (bukan aturan keras, supaya realistis)
    risiko = (
        3.2 * sigmoid((keausan - 205) / 6)                      # tool aus
        + 3.0 * sigmoid((8.6 - delta) / 0.4) * sigmoid((1380 - rpm) / 25)  # panas tidak terbuang
        + 3.0 * (sigmoid((3500 - daya) / 150) + sigmoid((daya - 9000) / 150))  # daya ekstrem
        + 3.5 * sigmoid((strain - batas) / 400)                 # tool aus di bawah beban besar
        + 0.6 * (jam_servis / 720)                              # makin lama sejak servis
        + 0.25 * (shift == "malam")
    )
    p_gagal = sigmoid(risiko * K_RISIKO + B_RISIKO)
    gagal = (rng.random(n) < p_gagal).astype(int)

    # getaran naik menjelang gagal: sinyal yang "aneh tapi benar", jangan dibuang
    getaran = np.clip(rng.normal(2.6, 0.7, n) + 0.9 * risiko / 3 + 1.2 * gagal * rng.random(n), 0.3, None)

    return pd.DataFrame({
        "tipe_produk": tipe,
        "suhu_udara_K": suhu_udara.round(1),
        "suhu_proses_K": suhu_proses.round(1),
        "kecepatan_rpm": rpm.round(0).astype(int),
        "torsi_Nm": torsi.round(1),
        "keausan_tool_menit": keausan,
        "getaran_mm_s": getaran.round(2),
        "jam_sejak_servis": jam_servis.astype(float),
        "shift_operator": shift,
        "gagal_24jam": gagal,
    })


def kotori(rng: np.random.Generator, df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    n = len(df)

    def pilih(frac):
        return rng.choice(n, size=int(n * frac), replace=False)

    df.loc[pilih(0.004), "suhu_proses_K"] = 0.0      # sensor suhu mati
    df.loc[pilih(0.005), "torsi_Nm"] = -1.0          # sensor torsi error
    df.loc[pilih(0.03), "getaran_mm_s"] = np.nan     # data hilang
    df.loc[pilih(0.02), "jam_sejak_servis"] = np.nan
    # typo kategori: spasi dan huruf besar
    typo = {"malam": ["Malam ", "MALAM", " malam"], "pagi": ["Pagi", "pagi "], "siang": ["Siang", " siang"]}
    for i in pilih(0.04):
        asli = df.at[i, "shift_operator"]
        df.at[i, "shift_operator"] = rng.choice(typo[asli])
    return df


def main():
    rng = np.random.default_rng(SEED)
    df = kotori(rng, bersih(rng, N_BERSIH))
    dup = df.sample(N_DUPLIKAT, random_state=SEED)
    df = pd.concat([df, dup]).sample(frac=1, random_state=SEED).reset_index(drop=True)
    df.to_csv(OUT, index=False)
    print(f"{OUT.name}: {len(df)} baris ({df.duplicated().sum()} duplikat), "
          f"gagal {df.drop_duplicates()['gagal_24jam'].mean():.1%}")


if __name__ == "__main__":
    main()
