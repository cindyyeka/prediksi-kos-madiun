import base64
import io
import os
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from django.shortcuts import render
from groq import Groq

from .forms import KosForm

BASE_DIR = Path(__file__).resolve().parent.parent

model = joblib.load(BASE_DIR / "ml_model" / "model.pkl")
df = pd.read_csv(BASE_DIR / "data" / "kos_madiun.csv")


def _fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


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
    ax.set_ylabel("Harga (ribu Rp)", fontsize=9.5)
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
            f"Rp {tinggi:,.0f}k",
            ha="center",
            va="bottom",
            fontsize=8.5,
            fontweight="bold",
            color="#243431",
        )

    ax.set_title("Rata-Rata Harga Kos Berdasarkan Fasilitas AC", fontsize=11, fontweight="bold", pad=10)
    ax.set_ylabel("Rata-rata harga (ribu Rp)", fontsize=9.5)
    ax.set_ylim(0, max(nilai_bar) * 1.18 if max(nilai_bar) > 0 else 100)
    ax.grid(axis="y", linestyle=":", alpha=0.45)

    return _fig_to_base64(fig)


def hitung_statistik_dataset():
    total_data = len(df)
    rata_rata_harga = round(float(df["harga"].mean()))
    harga_min = round(float(df["harga"].min()))
    harga_max = round(float(df["harga"].max()))

    return {
        "total_data": total_data,
        "rata_rata_harga": f"Rp {rata_rata_harga:,} ribu",
        "rentang_harga": f"Rp {harga_min:,}k – Rp {harga_max:,}k",
    }


def tentukan_kategori_harga(prediksi):
    q1 = float(df["harga"].quantile(0.33))
    q2 = float(df["harga"].quantile(0.67))

    if prediksi < q1:
        return "Terjangkau / Murah", "badge-murah"
    elif prediksi <= q2:
        return "Menengah / Standar", "badge-sedang"
    return "Relatif Tinggi / Premium", "badge-mahal"


def _panggil_groq(prompt_user):
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return "Penjelasan AI belum aktif (GROQ_API_KEY belum diatur)."
    try:
        client = Groq(api_key=api_key)
        response = client.chat.completions.create(
            model="qwen/qwen3.8-27b",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Kamu asisten analisis data kos di Madiun. Jawab SANGAT SINGKAT "
                        "(maksimal 1-2 kalimat saja, tanpa poin/daftar, langsung ke inti) "
                        "dalam bahasa Indonesia."
                    ),
                },
                {"role": "user", "content": prompt_user},
            ],
            temperature=0.2,
            max_tokens=120,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Gagal memuat AI: {e}"


def buat_penjelasan_ai(hasil_prediksi=None, info_prediksi=None):
    punya_ac = df["ac"].astype(str).str.strip().str.lower() == "ya"
    rata_ac = round(float(df.loc[punya_ac, "harga"].mean()))
    rata_non_ac = round(float(df.loc[~punya_ac, "harga"].mean()))
    korelasi_jarak = round(float(df["jarak_km"].corr(df["harga"])), 2)
    min_jarak = round(float(df["jarak_km"].min()), 1)
    max_jarak = round(float(df["jarak_km"].max()), 1)
    rata_harga = round(float(df["harga"].mean()))
    min_harga = round(float(df["harga"].min()))
    max_harga = round(float(df["harga"].max()))

    konteks_pred = ""
    if hasil_prediksi is not None and info_prediksi:
        konteks_pred = (
            f" Prediksi saat ini Rp {hasil_prediksi}k untuk jarak {info_prediksi['jarak']} km, "
            f"rating {info_prediksi['rating']}, {info_prediksi['ac_label']}."
        )

    penjelasan_scatter = _panggil_groq(
        f"Data jarak {min_jarak}-{max_jarak} km, korelasi jarak dengan harga {korelasi_jarak}.{konteks_pred} "
        "Jelaskan hubungan jarak dan harga kos (serta prediksinya jika ada) dalam 1-2 kalimat singkat."
    )

    penjelasan_ac = _panggil_groq(
        f"Rata-rata harga kos dengan AC Rp {rata_ac}k dan tanpa AC Rp {rata_non_ac}k. "
        "Jelaskan perbedaan harga berdasarkan fasilitas AC dalam 1-2 kalimat singkat."
    )

    penjelasan_tabel = _panggil_groq(
        f"Tabel kos punya {len(df)} data, harga min Rp {min_harga}k, max Rp {max_harga}k, rata-rata Rp {rata_harga}k. "
        "Ringkas isi tabel data kos ini dalam 1-2 kalimat singkat."
    )

    return {
        "scatter": penjelasan_scatter,
        "ac": penjelasan_ac,
        "tabel": penjelasan_tabel,
    }


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