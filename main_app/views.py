import base64
import io
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import pandas as pd
from django.shortcuts import render
from groq import Groq

from .forms import KosForm

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "ml_model" / "model.pkl")
df = pd.read_csv(BASE_DIR / "data" / "kos_madiun.csv")


# ---------- fungsi bantu format ----------
def _ke_rupiah_penuh(nilai_ribu):
    return int(round(float(nilai_ribu) * 1000))


def _format_rp(nilai_ribu):
    rupiah = _ke_rupiah_penuh(nilai_ribu)
    return f"Rp {rupiah:,}".replace(",", ".")


def _format_tick_rp(val, pos):
    rupiah = _ke_rupiah_penuh(val)
    return f"{rupiah:,}".replace(",", ".")


def _koma(angka, desimal=2):
    return str(round(float(angka), desimal)).replace(".", ",")


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


# ---------- grafik ----------
def buat_grafik_scatter(jarak=None, rating=None, ac_num=None, prediksi=None):
    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    ax.scatter(
        df["jarak_km"],
        df["harga"],
        alpha=0.75,
        color="#4f7c73",
        edgecolors="white",
        linewidths=0.6,
        label="Data kos",
    )

    if prediksi is not None:
        x_min = min(0.3, float(df["jarak_km"].min()), float(jarak))
        x_max = max(5.0, float(df["jarak_km"].max()), float(jarak))

        garis_x = pd.DataFrame({
            "jarak_km": [x_min, x_max],
            "rating": [rating, rating],
            "ac_num": [ac_num, ac_num],
        })
        garis_y = model.predict(garis_x)
        ax.plot(
            garis_x["jarak_km"],
            garis_y,
            color="#c85a48",
            linestyle="--",
            linewidth=1.8,
            label="Garis model",
        )
        ax.scatter(
            [jarak],
            [prediksi],
            color="#2b6cb0",
            s=140,
            marker="*",
            zorder=5,
            label="Prediksi kamu",
        )

    ax.set_title("Harga Kos vs Jarak ke Kampus", fontsize=11, fontweight="bold", pad=10)
    ax.set_xlabel("Jarak (km)", fontsize=9.5)
    ax.set_ylabel("Harga (Rp)", fontsize=9.5)
    ax.yaxis.set_major_formatter(FuncFormatter(_format_tick_rp))
    ax.grid(True, linestyle=":", alpha=0.45)
    ax.legend(fontsize=8.5, frameon=True)

    return _fig_to_base64(fig)


def buat_grafik_ac(ac_num=None):
    punya_ac = df["ac"].astype(str).str.strip().str.lower() == "ya"

    rata_non_ac = float(df.loc[~punya_ac, "harga"].mean())
    rata_dengan_ac = float(df.loc[punya_ac, "harga"].mean())

    label_bar = ["Tanpa AC", "Dengan AC"]
    nilai_bar = [rata_non_ac, rata_dengan_ac]
    warna_bar = ["#9ab5ae", "#3f6e65"]

    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    bars = ax.bar(label_bar, nilai_bar, color=warna_bar, width=0.48, edgecolor="none")

    if ac_num in (0, 1):
        bars[ac_num].set_edgecolor("#1f3d37")
        bars[ac_num].set_linewidth(2)

    for bar in bars:
        tinggi = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            tinggi + (max(nilai_bar) * 0.02 if max(nilai_bar) > 0 else 5),
            _format_rp(tinggi),
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#243431",
        )

    ax.set_title("Rata-Rata Harga Kos Berdasarkan Fasilitas AC", fontsize=11, fontweight="bold", pad=10)
    ax.set_ylabel("Rata-rata harga (Rp)", fontsize=9.5)
    ax.yaxis.set_major_formatter(FuncFormatter(_format_tick_rp))
    ax.set_ylim(0, max(nilai_bar) * 1.18 if max(nilai_bar) > 0 else 100)
    ax.grid(axis="y", linestyle=":", alpha=0.45)

    return _fig_to_base64(fig)


# ---------- statistik dan kategori ----------
def hitung_statistik_dataset():
    total_data = len(df)
    rata_rata_harga = float(df["harga"].mean())
    harga_min = float(df["harga"].min())
    harga_max = float(df["harga"].max())

    return {
        "total_data": total_data,
        "rata_rata_harga": _format_rp(rata_rata_harga),
        "rentang_harga": f"{_format_rp(harga_min)} – {_format_rp(harga_max)}",
    }


def tentukan_kategori_harga(prediksi):
    q1 = float(df["harga"].quantile(0.33))
    q2 = float(df["harga"].quantile(0.67))

    if prediksi < q1:
        return "Terjangkau / Murah", "badge-murah"
    elif prediksi <= q2:
        return "Menengah / Standar", "badge-sedang"
    return "Relatif Tinggi / Premium", "badge-mahal"


# ---------- penjelasan grafik ----------
def _kekuatan_korelasi(r):
    a = abs(r)
    if a < 0.2:
        kekuatan = "sangat lemah"
    elif a < 0.4:
        kekuatan = "lemah"
    elif a < 0.6:
        kekuatan = "sedang"
    else:
        kekuatan = "kuat"
    arah = "negatif" if r < 0 else "positif"
    return kekuatan, arah


def _panggil_groq(prompt_user, cadangan, wajib=()):
    """Panggil AI. Jika gagal, tidak diawali 'Grafik ini', atau tidak memuat
    kata/angka yang wajib ada, pakai kalimat cadangan buatan kode."""
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return cadangan
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Kamu pembaca grafik data kos. Tulis 1-2 kalimat singkat "
                        "bahasa Indonesia yang murni menjelaskan isi dan posisi data "
                        "pada grafik sesuai input yang diberikan. Kalimat pertama WAJIB "
                        "diawali dengan kata 'Grafik ini'. Jika fakta memuat bagian "
                        "'kesimpulan', tuliskan kesimpulan itu apa adanya. Jika ada "
                        "angka korelasi, tulis angkanya beserta tingkat kekuatannya. "
                        "Jika membandingkan dengan rata-rata sebuah kelompok, sebut nama "
                        "kelompok dan nominal rata-ratanya secara lengkap, jangan hanya "
                        "menulis 'kelompoknya'. Jangan pakai kalimat komentar, opini, "
                        "atau saran. Susun kalimat dengan redaksi alami yang tidak "
                        "meniru contoh. Tanpa poin/daftar dan tanpa menyebut nama kota "
                        "atau lokasi. Gunakan HANYA angka yang diberikan. Tulis nominal "
                        "uang dalam format rupiah penuh dengan titik pemisah seperti "
                        "Rp 1.067.857, dan JANGAN pakai kata 'ribu', 'rb', 'k', atau "
                        "'juta'."
                    ),
                },
                {"role": "user", "content": prompt_user},
            ],
            temperature=0.2,
            max_tokens=130,
        )
        teks = response.choices[0].message.content.strip()
        if not teks.lower().startswith("grafik ini"):
            return cadangan
        for kata in wajib:
            if kata.lower() not in teks.lower():
                return cadangan
        return teks
    except Exception:
        return cadangan


def buat_penjelasan_ai(hasil_prediksi=None, info_prediksi=None):
    harga = df["harga"]
    jumlah = len(df)
    min_harga = _format_rp(harga.min())
    max_harga = _format_rp(harga.max())
    rata_harga_angka = float(harga.mean())
    rata_harga = _format_rp(rata_harga_angka)

    # --- grafik scatter (jarak vs harga): rating TIDAK disebut ---
    r = float(df["jarak_km"].corr(harga))
    kekuatan, arah = _kekuatan_korelasi(r)
    jarak_min = _koma(df["jarak_km"].min(), 1)
    jarak_max = _koma(df["jarak_km"].max(), 1)

    fakta_scatter = (
        f"{jumlah} kos, jarak {jarak_min}-{jarak_max} km, harga {min_harga} sampai "
        f"{max_harga}. Korelasi jarak dan harga {_koma(r)}, tergolong {kekuatan} "
        f"(arah {arah})."
    )
    cadangan_scatter = (
        f"Grafik ini menunjukkan {jumlah} kos berjarak {jarak_min}-{jarak_max} km "
        f"dengan korelasi jarak dan harga {_koma(r)}, tergolong {kekuatan}."
    )

    if hasil_prediksi is not None and info_prediksi:
        selisih_rata = hasil_prediksi - rata_harga_angka
        posisi = "di atas" if selisih_rata >= 0 else "di bawah"
        jarak_in = _koma(info_prediksi["jarak"], 1)
        fakta_scatter += (
            f" Posisi kos yang dicek pengguna: jarak {jarak_in} km dengan prediksi "
            f"{_format_rp(hasil_prediksi)}, yaitu {posisi} rata-rata data "
            f"({rata_harga}) sebesar {_format_rp(abs(selisih_rata))}."
        )
        cadangan_scatter = (
            f"Grafik ini menunjukkan korelasi jarak dan harga {_koma(r)}, tergolong "
            f"{kekuatan}, dengan kos pada jarak {jarak_in} km diprediksi "
            f"{_format_rp(hasil_prediksi)}, {posisi} rata-rata data ({rata_harga})."
        )

    penjelasan_scatter = _panggil_groq(
        "Grafik scatter jarak vs harga. Fakta: " + fakta_scatter +
        " Jelaskan isi grafik (sertakan angka korelasi beserta tingkat kekuatannya), "
        "dan jika ada kos yang dicek, sebutkan posisinya terhadap rata-rata. "
        "Jangan menyebut rating.",
        cadangan_scatter,
        wajib=(kekuatan,),
    )

    # --- grafik batang AC: kelompok disebut lengkap + kesimpulan ---
    punya_ac = df["ac"].astype(str).str.strip().str.lower() == "ya"
    rata_ac_angka = float(harga[punya_ac].mean())
    rata_non_angka = float(harga[~punya_ac].mean())
    selisih = rata_ac_angka - rata_non_angka
    persen = abs(selisih) / rata_non_angka * 100
    lebih = "lebih mahal" if selisih >= 0 else "lebih murah"
    kesimpulan_ac = (
        "selisihnya kecil sehingga AC hampir tidak memengaruhi harga"
        if persen < 5
        else "selisihnya cukup besar sehingga AC memengaruhi harga"
    )

    fakta_ac = (
        f"rata-rata kos dengan AC {_format_rp(rata_ac_angka)}, rata-rata kos tanpa AC "
        f"{_format_rp(rata_non_angka)}, kos ber-AC {lebih} {_format_rp(abs(selisih))} "
        f"({_koma(persen, 1)}%). Kesimpulan: {kesimpulan_ac}."
    )
    cadangan_ac = (
        f"Grafik ini menunjukkan kos ber-AC {lebih} {_format_rp(abs(selisih))} "
        f"({_koma(persen, 1)}%) dari kos tanpa AC, {kesimpulan_ac}."
    )
    wajib_ac = ()

    if hasil_prediksi is not None and info_prediksi:
        dengan_ac = info_prediksi["ac_label"] == "Dengan AC"
        nama_kelompok = "kos dengan AC" if dengan_ac else "kos tanpa AC"
        rata_kelompok = rata_ac_angka if dengan_ac else rata_non_angka
        posisi_ac = "di atas" if hasil_prediksi >= rata_kelompok else "di bawah"
        fakta_ac += (
            f" Posisi kos yang dicek pengguna: prediksi {_format_rp(hasil_prediksi)}, "
            f"{posisi_ac} rata-rata {nama_kelompok} yang sebesar "
            f"{_format_rp(rata_kelompok)}."
        )
        cadangan_ac = (
            f"Grafik ini menunjukkan {kesimpulan_ac}, dengan prediksi "
            f"{_format_rp(hasil_prediksi)} berada {posisi_ac} rata-rata "
            f"{nama_kelompok} sebesar {_format_rp(rata_kelompok)}."
        )
        wajib_ac = (_format_rp(rata_kelompok),)

    penjelasan_ac = _panggil_groq(
        "Grafik batang rata-rata harga berdasarkan AC. Fakta: " + fakta_ac +
        " Jelaskan isi grafik, tuliskan kesimpulannya, dan jika ada kos yang dicek, "
        "sebutkan posisinya terhadap rata-rata kelompok lengkap dengan nama "
        "kelompok dan nominalnya.",
        cadangan_ac,
        wajib=wajib_ac,
    )

    # --- ringkasan data ---
    cadangan_tabel = (
        f"Grafik ini menunjukkan {jumlah} kos dengan harga {min_harga} sampai "
        f"{max_harga}, rata-rata {rata_harga}."
    )
    penjelasan_tabel = _panggil_groq(
        f"Ringkasan data kos. Fakta: {jumlah} kos, harga terendah {min_harga}, "
        f"tertinggi {max_harga}, rata-rata {rata_harga}. Rangkum gambaran umum harganya.",
        cadangan_tabel,
    )

    return {
        "scatter": penjelasan_scatter,
        "ac": penjelasan_ac,
        "tabel": penjelasan_tabel,
    }


# ---------- halaman utama ----------
def index(request):
    hasil = None
    info_prediksi = None
    ac_num = None

    if request.method == "POST":
        form = KosForm(request.POST)
        if form.is_valid():
            jarak = form.cleaned_data["jarak_km"]
            rating = form.cleaned_data["rating"]
            ac_raw = form.cleaned_data["ac"]
            ac_num = 1 if str(ac_raw).strip().lower() == "ya" else 0

            data_baru = pd.DataFrame({
                "jarak_km": [jarak],
                "rating": [rating],
                "ac_num": [ac_num],
            })
            hasil = round(float(model.predict(data_baru)[0]))
            kategori, badge_class = tentukan_kategori_harga(hasil)

            info_prediksi = {
                "jarak": jarak,
                "rating": rating,
                "ac_label": "Dengan AC" if ac_num == 1 else "Tanpa AC",
                "kategori": kategori,
                "badge_class": badge_class,
            }
    else:
        form = KosForm()

    if hasil is not None:
        grafik_scatter = buat_grafik_scatter(
            jarak=info_prediksi["jarak"],
            rating=info_prediksi["rating"],
            ac_num=ac_num,
            prediksi=hasil,
        )
    else:
        grafik_scatter = buat_grafik_scatter()

    grafik_ac = buat_grafik_ac(ac_num=ac_num)
    statistik = hitung_statistik_dataset()
    penjelasan_ai = buat_penjelasan_ai(hasil, info_prediksi)

    return render(
        request,
        "main_app/index.html",
        {
            "form": form,
            "hasil": hasil,
            "info_prediksi": info_prediksi,
            "grafik_scatter": grafik_scatter,
            "grafik_ac": grafik_ac,
            "statistik": statistik,
            "penjelasan_ai": penjelasan_ai,
        },
    )