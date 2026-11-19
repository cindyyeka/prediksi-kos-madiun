# Prediksi Harga Kos Madiun

Aplikasi web berbasis **Django** dan model regresi machine learning untuk memprediksi estimasi harga sewa kos bulanan di sekitar kampus di Madiun berdasarkan jarak ke kampus, rating kos, dan ketersediaan fasilitas AC, lengkap dengan visualisasi grafik dan penjelasan singkat berbasis AI.

## Fitur Aplikasi
- Ringkasan statistik dataset (total data kos, rata-rata harga, dan rentang harga minimum–maksimum).
- Form input karakteristik kos (`jarak_km`, `rating`, dan fasilitas `ac`).
- Hasil prediksi harga sewa per bulan beserta kategori tingkat harga.
- Visualisasi *scatter plot* harga kos terhadap jarak ke kampus (beserta garis model dan titik prediksi pengguna) serta diagram batang rata-rata harga berdasarkan fasilitas AC, lengkap dengan penjelasan singkat AI.

## Struktur Folder
- `manage.py` & `core_proyek/` — Konfigurasi utama proyek Django.
- `main_app/` — Logika aplikasi (`views.py`, `forms.py`, `urls.py`) dan template tampilan (`templates/main_app/index.html`).
- `train_model.py` & `ml_model/model.pkl` — Skrip pelatihan model dan file model regresi terlatih.
- `data/kos_madiun.csv` — Dataset harga dan spesifikasi kos.
- `Procfile`, `.gitignore`, & `requirements.txt` — Konfigurasi deployment dan daftar dependensi.

## Dataset, Sumber, dan Lisensi
- **File Dataset:** `data/kos_madiun.csv` (30 data kos).
- **Sumber:** Dataset modul praktikum mata kuliah Sistem Cerdas.
- **Lisensi:** Digunakan untuk pembelajaran dan tugas akademik.