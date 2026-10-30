# Machine Learning · MKB UPI — praktikum

Bahan hands-on seri kuliah praktisi **From notebook to production**.

## P2 · Act — dari data sensor ke model artifact

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nurhikam/mkb-ml-p2/blob/main/notebooks/P2_Act_STUDENT.ipynb)

1. Klik tombol **Open in Colab** di atas.
2. **File → Save a copy in Drive**, supaya kalian kerja di salinan sendiri.
3. Jalankan dari atas, isi 8 TODO. Tiap TODO punya sel ✅ Cek.
4. Mentok atau ketinggalan? Buka **versi jawaban** di bawah, salin isi sel TODO-nya ke notebook kalian.
5. Di akhir, unduh `artifact_p2.zip` (berisi `model.joblib`, `metadata.json`, `src/features.py`) dan bawa ke P3.

**Versi jawaban** (semua TODO sudah terisi):
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/nurhikam/mkb-ml-p2/blob/main/notebooks/P2_Act_JAWABAN.ipynb)

**Kasus:** prediksi mesin CNC yang akan gagal dalam 24 jam (predictive maintenance), pabrik komponen otomotif.

## Isi repo

| Path | Isi |
|---|---|
| `notebooks/P2_Act_STUDENT.ipynb` | notebook hands-on P2 (TODO kosong) |
| `notebooks/P2_Act_JAWABAN.ipynb` | notebook yang sama, semua TODO sudah terisi |
| `data/mesin_cnc.csv` | 10,150 baris data sensor sintetis (sengaja kotor) |
| `data/generate.py` | skrip pembuat dataset (`python data/generate.py`) |

Datanya **sintetis**: data pabrik asli hampir selalu rahasia. Strukturnya meniru data sensor mesin perkakas
(terinspirasi dataset AI4I 2020) dan sengaja dikotori: duplikat, nilai sensor mustahil, nilai kosong, typo kategori.

Mentok? Tanya di QnA kelas dan tempel pesan error lengkapnya.
