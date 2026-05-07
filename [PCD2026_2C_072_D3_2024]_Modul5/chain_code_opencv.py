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
        # else: # Kasus anomali (perpindahan > 1 piksel?): bisa di-log jika perlu
            # print(f"Peringatan: Perpindahan non-standar ({dx},{dy}) dari {p1} ke {p2}")

    return chain_code

# --- Alur Proses Utama ---
if __name__ == "__main__":
    # 1. Pemuatan Citra (langsung grayscale)
    img_path = 'tangram1.jpg'
    try:
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
        if img is None:
            raise FileNotFoundError(f"Citra tidak ditemukan di: {img_path}")

        # 2. Binarisasi (Sesuaikan threshold & type berdasarkan citra Anda)
        # Contoh: objek gelap di latar terang -> pakai THRESH_BINARY_INV
        threshold_value = 127 
        _, binary_img = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY_INV) 
        # Jika objek terang di latar gelap -> pakai THRESH_BINARY

        # 3. Deteksi Kontur (Wajib CHAIN_APPROX_NONE)
        contours, hierarchy = cv2.findContours(binary_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

        # 4. Proses Kontur Terbesar (Contoh)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Visualisasi (Opsional)
            img_display = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) # Konversi untuk warna
            cv2.drawContours(img_display, [largest_contour], -1, (0, 255, 0), 1) # Gambar kontur hijau

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
    except Exception as e:
        print(f"Error: {e}")
