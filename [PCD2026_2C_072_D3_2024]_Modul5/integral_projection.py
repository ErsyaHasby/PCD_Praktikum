import cv2
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # 1. Pemuatan Citra (langsung grayscale)
    # Ganti dengan path citra teks/objek Anda. Anda bisa coba 'pesawat.jpg' atau 'tangram1.jpg' lagi.
    img_path = 'pesawat.jpg' 
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Citra tidak ditemukan di: {img_path}")
        exit()

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
