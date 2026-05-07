# Modul 5 : Kontur dan Fitur Bentuk Citra via OpenCV

**Abstrak**

Modul ini menyajikan pembahasan mendalam mengenai teknik-teknik fundamental dalam analisis kontur dan ekstraksi fitur bentuk pada citra digital menggunakan pustaka OpenCV dalam lingkungan Python. Fokus utama adalah pada tiga metodologi inti: pembentukan Kode Rantai (Chain Code) untuk representasi kontur, penggunaan deteksi tepi Canny sebagai pendekatan untuk konsep Kode Retakan (Crack Code), dan kalkulasi fitur bentuk global melalui Proyeksi Integral. Materi ini dirancang untuk memberikan pemahaman konseptual, implementasi praktis, dan panduan interpretasi bagi pengembang yang ingin memanfaatkan informasi bentuk dalam aplikasi visi komputer.

## 1. Prasyarat Pustaka dan Instalasi

Sebelum melanjutkan, pastikan lingkungan Python Anda memiliki pustaka-pustaka berikut terinstal. Gunakan pip (atau manajer paket pilihan Anda) untuk instalasi:

```bash
pip install opencv-python numpy matplotlib
```

- **opencv-python:** Pustaka inti OpenCV untuk fungsi-fungsi pemrosesan citra.
- **numpy:** Pustaka fundamental untuk komputasi numerik, khususnya operasi array yang efisien.
- **matplotlib:** Pustaka untuk visualisasi data, digunakan di sini untuk menampilkan citra dan plot proyeksi.

---

## 2. Representasi Kontur Menggunakan Kode Rantai (Chain Code)

### 2.1. Konsep Dasar

Kode Rantai (Chain Code), khususnya Kode Rantai Freeman 8-arah, adalah metode representasi batas (kontur) objek digital yang mengkodekan kontur sebagai serangkaian vektor perpindahan antar piksel tetangga yang berurutan. Standar 8-arah mengkodekan perpindahan ke salah satu dari delapan piksel tetangga menggunakan digit 0-7.

### 2.2. Ilustrasi Konseptual

*(Ilustrasi Kode Rantai Freeman 8-arah)*

### 2.3. Proses Implementasi

1. **Binarisasi Citra:** Transformasi citra input menjadi format biner.
2. **Ekstraksi Kontur:** Gunakan `cv2.findContours` dengan `method=cv2.CHAIN_APPROX_NONE`.
3. **Inisialisasi:** Pilih titik awal pada kontur.
4. **Penelusuran dan Pengkodean:** Iterasi titik kontur, hitung `(dx, dy)` antar titik berturutan, tentukan kode arah, catat kode.
5. **Terminasi:** Ulangi hingga semua titik ditelusuri.

### 2.4. Prasyarat Input dan Interpretasi Output

- **Prasyarat Input:** Citra biner (tipe data `uint8`), di mana objek memiliki nilai piksel yang sama (misal, 255) dan berbeda dari latar belakang (misal, 0). Kualitas binarisasi sangat mempengaruhi hasil.
- **Parameter Kunci `findContours`:** `method=cv2.CHAIN_APPROX_NONE` **wajib** digunakan untuk memastikan semua titik piksel pada kontur diekstraksi, memungkinkan pergerakan 1-piksel yang diperlukan untuk kode rantai. `cv2.CHAIN_APPROX_SIMPLE` akan menghilangkan titik-titik di tengah segmen garis lurus, merusak urutan kode rantai.
- **Interpretasi Output:** Sebuah list Python berisi integer dari 0 hingga 7, merepresentasikan urutan arah pergerakan sepanjang kontur. Panjang list ini kira-kira sama dengan panjang perimeter kontur dalam piksel.

### 2.5. Fungsi OpenCV Relevan

- `cv2.threshold()`
- `cv2.findContours()`
- `cv2.drawContours()`

### 2.6. Contoh Implementasi

*(Pastikan opencv-python sudah terinstall, untuk code ini jangan gunakan opencv-python-headless)*

```python
import cv2
import numpy as np

def generate_freeman_chain_code(contour):
    """
    Menghasilkan Kode Rantai Freeman 8-arah dari kontur OpenCV.
    ASUMSI: kontur didapat dari findContours dengan CHAIN_APPROX_NONE.
    """
    chain_code = []
    if len(contour) < 2:
        return chain_code # Kontur harus punya minimal 2 titik

    # Pemetaan (dx, dy) ke kode arah Freeman (sumbu Y positif ke bawah)
    directions = {
        (1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3,
        (-1, 0): 4, (-1, -1): 5, (0, -1): 6, (1, -1): 7
    }

    for i in range(len(contour)):
        p1 = contour[i][0] # Titik saat ini (format: [[x, y]])
        # Dapatkan titik berikutnya, gunakan modulo % untuk kembali ke titik awal 
        # pada iterasi terakhir (menangani kontur tertutup).
        p2 = contour[(i + 1) % len(contour)][0] 

        dx = p2[0] - p1[0] # Perbedaan X
        dy = p2[1] - p1[1] # Perbedaan Y (Ingat: Y positif ke bawah)

        # Dengan CHAIN_APPROX_NONE, dx/dy harusnya hanya -1, 0, atau 1.
        # Normalisasi sign memastikan ini, meskipun secara teori tidak perlu.
        norm_dx = np.sign(dx) 
        norm_dy = np.sign(dy)
        
        # Cari kode arah dari dictionary berdasarkan perpindahan (dx, dy)
        code = directions.get((norm_dx, norm_dy)) 
        if code is not None:
            chain_code.append(code)

    return chain_code

# --- Alur Proses Utama ---
# 1. Pemuatan Citra (langsung grayscale)
img_path = 'WINWORD_GND0i7aMnp.png'
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
if img is None:
    raise FileNotFoundError(f"Citra tidak ditemukan di: {img_path}")

# 2. Binarisasi (Sesuaikan threshold & type berdasarkan citra Anda)
threshold_value = 127 
_, binary_img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY_INV) 

# 3. Deteksi Kontur (Wajib CHAIN_APPROX_NONE)
contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# 4. Proses Kontur Terbesar (Contoh)
if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Visualisasi (Opsional)
    img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) 
    cv2.drawContours(img_display, [largest_contour], -1, (0, 255, 0), 1) 

    # Generasi Kode Rantai
    chain_code_result = generate_freeman_chain_code(largest_contour)
    
    print(f"Jumlah Kontur Ditemukan: {len(contours)}")
    print(f"Kode Rantai Kontur Terbesar (Panjang {len(chain_code_result)}):")
    print(chain_code_result)

    # Tampilkan hasil
    cv2.imshow("Citra Asli", img)
    cv2.imshow("Citra Biner", binary_img)
    cv2.imshow("Kontur Terdeteksi", img_display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
else:
    print("Tidak ada kontur yang terdeteksi.")
```

#### Alternatif Code dengan `plt.show`

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

def generate_freeman_chain_code(contour):
    """
    Menghasilkan Kode Rantai Freeman 8-arah dari kontur OpenCV.
    ASUMSI: kontur didapat dari findContours dengan CHAIN_APPROX_NONE.
    """
    chain_code = []
    if len(contour) < 2:
        return chain_code

    directions = {
        (1, 0): 0, (1, 1): 1, (0, 1): 2, (-1, 1): 3,
        (-1, 0): 4, (-1, -1): 5, (0, -1): 6, (1, -1): 7
    }

    for i in range(len(contour)):
        p1 = contour[i][0]
        p2 = contour[(i + 1) % len(contour)][0]

        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]

        norm_dx = np.sign(dx)
        norm_dy = np.sign(dy)

        code = directions.get((norm_dx, norm_dy))
        if code is not None:
            chain_code.append(code)

    return chain_code

# --- Alur Proses Utama ---
img_path = 'path/ke/gambar_bentuk_sederhana.png' # <--- GANTI PATH INI
try:
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Citra tidak ditemukan atau tidak dapat dibaca di: {img_path}")
except Exception as e:
    print(f"Error saat memuat citra: {e}")
    print("Pastikan path citra sudah benar dan file citra tidak rusak.")
    exit()

threshold_value = 127
_, binary_img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY_INV)

contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

# --- Persiapan Visualisasi dengan Matplotlib ---
fig, axs = plt.subplots(2, 2, figsize=(10, 8))

axs[0, 0].imshow(img, cmap='gray')
axs[0, 0].set_title('Citra Asli (Grayscale)')
axs[0, 0].axis('off')

axs[0, 1].imshow(binary_img, cmap='gray')
axs[0, 1].set_title('Citra Biner (Hasil Threshold)')
axs[0, 1].axis('off')

chain_code_str = "Tidak ada kontur ditemukan."
img_contour_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

if contours:
    largest_contour = max(contours, key=cv2.contourArea)
    img_contour_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(img_contour_display, [largest_contour], -1, (0, 255, 0), 1)

    chain_code_result = generate_freeman_chain_code(largest_contour)
    
    max_line_len = 70
    wrapped_code = ""
    current_line_len = 0
    for i, code_str in enumerate(map(str, chain_code_result)):
        item = code_str + (", " if i < len(chain_code_result) - 1 else "")
        if current_line_len + len(item) > max_line_len:
            wrapped_code += "\n"
            current_line_len = 0
        wrapped_code += item
        current_line_len += len(item)

    chain_code_str = (
        f"Jumlah Kontur Total: {len(contours)}\n"
        f"Kode Rantai Kontur Terbesar (Panjang {len(chain_code_result)}):\n"
        f"{wrapped_code}"
    )

img_rgb_display = cv2.cvtColor(img_contour_display, cv2.COLOR_BGR2RGB)
axs[1, 0].imshow(img_rgb_display)
axs[1, 0].set_title('Kontur Terbesar Terdeteksi')
axs[1, 0].axis('off')

axs[1, 1].axis('off')
axs[1, 1].text(0.05, 0.95, chain_code_str, ha='left', va='top', fontsize=9, wrap=True)
axs[1, 1].set_title('Hasil Kode Rantai')

plt.tight_layout(pad=1.5)
plt.suptitle("Analisis Kode Rantai", fontsize=16)
plt.subplots_adjust(top=0.92)
plt.show()
```

### 2.7. Penjelasan Kode

- Fungsi `generate_freeman_chain_code` mengimplementasikan logika inti: iterasi titik, kalkulasi `dx, dy`, pemetaan ke kode 0-7, dan penyimpanan.
- Pemuatan, binarisasi (perhatikan pemilihan `THRESH_BINARY` vs `THRESH_BINARY_INV`), dan deteksi kontur (`CHAIN_APPROX_NONE` krusial) dilakukan sebagai prasyarat.
- Kontur terbesar dipilih sebagai contoh, dan kode rantainya dihasilkan serta dicetak.

### 2.8. Tuning Parameter dan Penanganan Masalah Umum

- **Thresholding:** Nilai threshold (`threshold_value`) adalah parameter paling kritis. Jika objek tidak terpisah sempurna dari latar belakang, kontur akan salah atau noise akan masuk. Coba gunakan `cv2.THRESH_OTSU` untuk thresholding otomatis jika kontras baik. Pastikan `THRESH_BINARY` atau `THRESH_BINARY_INV` dipilih dengan benar agar objek target bernilai 255.
- **Noise:** Noise pada batas objek akan menghasilkan segmen kode rantai yang tidak relevan dan memperpanjang kode. Pertimbangkan pra-pemrosesan dengan filter blur (misal, `cv2.GaussianBlur`) *sebelum* thresholding jika noise signifikan.
- **Kontur Kosong:** Jika tidak ada kontur terdeteksi (`contours` kosong), periksa hasil binarisasi. Apakah objek benar-benar putih (255) dan latar belakang hitam (0)? Apakah threshold terlalu ekstrim?
- **Lupa `CHAIN_APPROX_NONE`:** Jika menggunakan `CHAIN_APPROX_SIMPLE`, fungsi `generate_freeman_chain_code` kemungkinan akan gagal atau menghasilkan hasil aneh karena perpindahan antar titik bisa lebih dari 1 piksel (`dx` atau `dy` > 1).

### 2.9. Aplikasi Umum

- Pengenalan Karakter Optik (OCR)
- Analisis dan Klasifikasi Bentuk
- Pencocokan Template Berbasis Kontur
- Kompresi Data Representasi Bentuk

### 2.10. Analisis Kritis

- **Kelebihan:** Representasi kompak; Invarian translasi; Dasar analisis geometris.
- **Kekurangan:** Sensitif noise & binarisasi; Tidak invarian rotasi/skala; Titik awal mempengaruhi urutan; Tidak disediakan langsung oleh OpenCV.

---

## 3. Deteksi Tepi (Canny) sebagai Pendekatan Kode Retakan (Crack Code)

### 3.1. Konsep Dasar

Kode Retakan (Crack Code) merepresentasikan batas *antar* piksel. Algoritma Canny (`cv2.Canny`) mendeteksi diskontinuitas intensitas tinggi, yang sering berlokasi di batas objek. Hasil Canny (peta piksel tepi) dapat dianggap sebagai **lokasi potensial** dari “retakan” ini, berfungsi sebagai *proxy* atau *input* untuk algoritma pelacakan retakan yang lebih kompleks (yang tidak dibahas di sini). **Penting:** Canny *tidak* menghasilkan urutan kode retakan secara langsung.

### 3.2. Algoritma Canny 

1. Reduksi Noise (Gaussian Blur)
2. Kalkulasi Gradien Intensitas (Sobel)
3. Non-Maximum Suppression
4. Thresholding Hysteresis

### 3.3 Ilustrasi Canny

*(Ilustrasi Algoritma Canny dan Hasilnya)*

### 3.4. Prasyarat Input dan Interpretasi Output

- **Prasyarat Input:** Citra grayscale (tipe data `uint8`). Pra-pemrosesan dengan `cv2.GaussianBlur` sangat direkomendasikan.
- **Parameter Kunci:** `threshold1` (ambang bawah) dan `threshold2` (ambang atas) untuk histeresis.
- **Interpretasi Output:** Citra biner (`uint8`) dengan ukuran sama seperti input, di mana piksel putih (255) menandakan lokasi tepi yang terdeteksi, dan piksel hitam (0) adalah non-tepi.

### 3.5. Fungsi OpenCV Relevan

- `cv2.cvtColor()`
- `cv2.GaussianBlur()`
- `cv2.Canny()`

### 3.6. Contoh Implementasi

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. Pemuatan Citra
img_path = 'cameraman.png'
img = cv2.imread(img_path)
if img is None:
    raise FileNotFoundError(f"Citra tidak ditemukan di: {img_path}")

# 2. Konversi ke Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 3. Reduksi Noise (Langkah pra-pemrosesan penting!)
# Eksperimen dengan ukuran kernel (ksize) dan sigma
blurred = cv2.GaussianBlur(gray, (5, 5), 0) # Kernel 5x5 umum digunakan

# 4. Deteksi Tepi Canny
# Nilai threshold SANGAT bergantung pada citra. Perlu eksperimen!
low_threshold = 50
high_threshold = 150 # Aturan umum: high sekitar 2x-3x low
edges = cv2.Canny(blurred, low_threshold, high_threshold)

# 5. Visualisasi Hasil
plt.figure(figsize=(12, 6))
plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)) # Tampilkan citra asli (konversi warna BGR->RGB)
plt.title('Citra Asli')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(blurred, cmap='gray') # Tampilkan hasil blur
plt.title('Grayscale + Gaussian Blur')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(edges, cmap='gray') # Tampilkan peta tepi hasil Canny
plt.title(f'Tepi Canny (Th={low_threshold},{high_threshold})')
plt.axis('off')

plt.tight_layout()
plt.show()
```

### 3.7. Penjelasan Kode

- Citra dimuat, dikonversi ke grayscale, dan dihaluskan dengan Gaussian Blur.
- `cv2.Canny` dipanggil dengan citra hasil blur dan dua ambang batas.
- Hasil (peta tepi biner) divisualisasikan bersama citra asli dan hasil blur untuk perbandingan.

### 3.8. Tuning Parameter dan Penanganan Masalah Umum

- **Ambang Batas (`threshold1`, `threshold2`):** Ini adalah parameter *paling* penting untuk Canny.
    - Jika **terlalu banyak tepi/noise** terdeteksi: Naikkan `threshold2` (dan `threshold1` secara proporsional, misal tetap 1:2 atau 1:3). Atau, tingkatkan `GaussianBlur` (kernel lebih besar atau sigma > 0).
    - Jika **tepi penting hilang**: Turunkan `threshold1` (dan `threshold2` secara proporsional). Pastikan `GaussianBlur` tidak terlalu kuat sehingga menghapus detail tepi.
    - **Eksperimen:** Mulai dengan rasio 1:2 atau 1:3 (misal, 50:150, 70:140, 100:200) dan sesuaikan berdasarkan hasil visual pada citra spesifik Anda.
- **Gaussian Blur:** Ukuran kernel (harus ganjil) dan sigma mengontrol tingkat penghalusan. Blur yang terlalu kuat dapat menghilangkan tepi lemah. Blur yang tidak cukup dapat membiarkan noise terdeteksi sebagai tepi. Kernel (5,5) dengan sigma 0 adalah titik awal yang baik.

### 3.9. Aplikasi Umum (Deteksi Tepi)

- Segmentasi Citra, Ekstraksi Fitur, Analisis Tekstur
- Pencitraan Medis, Inspeksi Industri

### 3.10. Analisis Kritis

- **Kelebihan (Canny):** Deteksi tepi baik, reduksi noise internal, tepi tipis, parameter fleksibel.
- **Kekurangan (Canny sebagai Proxy Crack Code):** Output adalah *peta piksel tepi*, bukan *kode urutan retakan*; Perlu algoritma pelacakan tambahan untuk kode retakan formal; Hasil sensitif terhadap pemilihan ambang batas; Lebih intensif komputasi daripada thresholding.

---

## 4. Kalkulasi Fitur Bentuk via Proyeksi Integral

### 4.1. Konsep Dasar

Proyeksi Integral (Proyeksi Histogram) mereduksi dimensi citra dengan menjumlahkan nilai piksel sepanjang baris atau kolom, menghasilkan profil 1D distribusi massa piksel. Sangat berguna pada citra **biner** (objek=1/putih, latar=0/hitam).

- **Proyeksi Horizontal:** Jumlah per kolom -> Profil Vertikal.
- **Proyeksi Vertikal:** Jumlah per baris -> Profil Horizontal.

### 4.2. Ilustrasi

*(Ilustrasi Proyeksi Integral pada Citra Teks)*

### 4.3. Prasyarat Input dan Interpretasi Output

- **Prasyarat Input:** Citra biner (`uint8`), di mana piksel objek target memiliki nilai **non-nol** (idealnya 1 atau 255) dan piksel latar belakang bernilai **nol**. Kesalahan dalam binarisasi (misal, objek hitam di latar putih) akan menghasilkan proyeksi yang tidak bermakna jika langsung dijumlahkan. Normalisasi ke 0 dan 1 (objek=1, latar=0) direkomendasikan.
- **Parameter Kunci `numpy.sum`:** `axis=0` untuk proyeksi horizontal (jumlah per kolom), `axis=1` untuk proyeksi vertikal (jumlah per baris).
- **Interpretasi Output:** Dua array NumPy 1D.
    - `horizontal_projection`: Panjangnya sama dengan lebar citra. Elemen ke-`j` adalah jumlah piksel objek di kolom `j`.
    - `vertical_projection`: Panjangnya sama dengan tinggi citra. Elemen ke-`i` adalah jumlah piksel objek di baris `i`.

### 4.4. Fungsi Relevan

- `cv2.threshold()`
- `numpy.sum()` (Metode utama dan paling efisien)
- `cv2.reduce()` (Alternatif OpenCV)

### 4.5. Contoh Implementasi

```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 1. Pemuatan Citra (langsung grayscale)
    img_path = 'tier1.png' # Ganti dgn path citra teks/objek
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Citra tidak ditemukan di: {img_path}")

    # 2. Binarisasi (KRUSIAL: Objek harus PUTIH/NON-NOL, Latar HITAM/NOL)
    # Gunakan Otsu untuk otomatisasi jika kontras baik
    # Jika teks hitam di latar putih, gunakan THRESH_BINARY_INV
    _, binary_img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU) 
    # Jika perlu, pastikan objek = 255, latar = 0. Jika terbalik: binary_img = 255 - binary_img

    # Normalisasi ke 0 dan 1 (Objek=1, Latar=0) untuk interpretasi mudah
    binary_norm = binary_img / 255.0 

    # 3. Proyeksi Horizontal (Sum per Kolom -> Profil Vertikal)
    # axis=0: menjumlahkan sepanjang dimensi baris (secara vertikal)
    horizontal_projection = np.sum(binary_norm, axis=0)

    # 4. Proyeksi Vertikal (Sum per Baris -> Profil Horizontal)
    # axis=1: menjumlahkan sepanjang dimensi kolom (secara horizontal)
    vertical_projection = np.sum(binary_norm, axis=1)

    # 5. Visualisasi Hasil (Layout ditingkatkan)
    height, width = binary_norm.shape

    # Buat figure dan axes dengan GridSpec untuk kontrol layout lebih baik
    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 2, width_ratios=(4, 1), height_ratios=(1, 4),
                          left=0.1, right=0.9, bottom=0.1, top=0.9,
                          wspace=0.05, hspace=0.05)

    # Axes untuk citra biner (pojok kiri bawah)
    ax_img = fig.add_subplot(gs[1, 0])
    ax_img.imshow(binary_norm, cmap='gray')
    ax_img.set_title('Citra Biner (Objek=1)')
    ax_img.set_xlabel('Indeks Kolom')
    ax_img.set_ylabel('Indeks Baris')

    # Axes untuk Proyeksi Horizontal (di atas citra biner)
    ax_hproj = fig.add_subplot(gs[0, 0], sharex=ax_img) # Bagikan sumbu X
    ax_hproj.plot(np.arange(width), horizontal_projection)
    ax_hproj.set_title('Proyeksi Horizontal (Profil Vertikal)')
    ax_hproj.set_ylabel('Jumlah Piksel')
    plt.setp(ax_hproj.get_xticklabels(), visible=False) # Sembunyikan label X

    # Axes untuk Proyeksi Vertikal (di kanan citra biner)
    ax_vproj = fig.add_subplot(gs[1, 1], sharey=ax_img) # Bagikan sumbu Y
    ax_vproj.plot(vertical_projection, np.arange(height)) 
    ax_vproj.set_title('Proyeksi Vertikal')
    ax_vproj.set_xlabel('Jumlah Piksel')
    ax_vproj.invert_yaxis() # Cocokkan orientasi Y citra
    plt.setp(ax_vproj.get_yticklabels(), visible=False) # Sembunyikan label Y

    plt.suptitle("Analisis Proyeksi Integral", fontsize=14)
    plt.show()
```

### 4.6. Penjelasan Kode

- Citra dibinarisasi (pastikan objek=1, latar=0) dan dinormalisasi.
- `np.sum` digunakan dengan `axis=0` dan `axis=1` untuk menghitung kedua proyeksi.
- Matplotlib `GridSpec` digunakan untuk membuat layout plot yang rapi, menempatkan plot proyeksi di atas dan di kanan citra biner, dengan sumbu yang dibagikan (`sharex`, `sharey`) untuk korelasi visual yang mudah.

### 4.7. Tuning Parameter dan Penanganan Masalah Umum

- **Binarisasi adalah Segalanya:** Hasil proyeksi *sepenuhnya* bergantung pada kualitas binarisasi. Pastikan objek target secara konsisten bernilai 1 (atau 255 sebelum normalisasi) dan latar belakang 0. Jika citra asli memiliki pencahayaan tidak merata, pertimbangkan teknik thresholding adaptif (`cv2.adaptiveThreshold`) sebelum proyeksi.
- **Objek Hitam?:** Jika setelah thresholding objek Anda hitam (0) dan latar putih (255), hasil `np.sum` tidak akan menunjukkan lokasi objek. Balikkan citra biner (`binary_img = 255 - binary_img`) sebelum normalisasi dan penjumlahan.
- **Profil “Rata”:** Jika profil proyeksi hampir rata atau nol, kemungkinan besar binarisasi gagal atau objek tidak ada dalam citra.

### 4.8. Aplikasi Umum

- Segmentasi Baris Teks/Karakter (OCR)
- Deteksi Lokasi Kasar Objek, Cropping Otomatis
- Analisis Pola Periodik (Barcode, Partitur)

### 4.9. Analisis Kritis

- **Kelebihan:** Sangat cepat, implementasi mudah, efektif untuk fitur global/repetitif sejajar sumbu.
- **Kekurangan:** Kehilangan info spasial detail, tidak efektif untuk bentuk kompleks/tumpang tindih, sensitif rotasi, sangat bergantung pada binarisasi.
