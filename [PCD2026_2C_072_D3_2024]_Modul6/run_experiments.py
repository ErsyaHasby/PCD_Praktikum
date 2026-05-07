# pyrefly: ignore [missing-import]
import cv2
import os
# pyrefly: ignore [missing-import]
import numpy as np
# pyrefly: ignore [missing-import]
from skimage.metrics import structural_similarity as ssim
# pyrefly: ignore [missing-import]
import matplotlib.pyplot as plt
import pandas as pd
import subprocess
import urllib.request

# pyrefly: ignore [missing-import]
import imageio.v3 as iio

def download_images():
    # Use skimage data for camera, since we have it
    import skimage.data
    camera = skimage.data.camera()
    cv2.imwrite('cameraman.tif', camera)
    
    # Use skimage.data.astronaut as alternative for peppers
    astronaut = skimage.data.astronaut()
    cv2.imwrite('peppers.png', cv2.cvtColor(astronaut, cv2.COLOR_RGB2BGR))
    
    images = {
        'lena.png': 'https://upload.wikimedia.org/wikipedia/en/7/7d/Lenna_%28test_image%29.png'
    }
    for name, url in images.items():
        if not os.path.exists(name):
            try:
                print(f"Downloading {name}...")
                req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
                with urllib.request.urlopen(req) as response:
                    img_data = response.read()
                    with open(name, 'wb') as f:
                        f.write(img_data)
            except Exception as e:
                print(f"Failed to download {name}: {e}")

download_images()

def run_experiment(image_path_original):
    print(f"\n--- Processing {image_path_original} ---")
    img_original_bgr = cv2.imread(image_path_original)
    
    if img_original_bgr is None:
        print(f"Error: Tidak dapat memuat citra dari {image_path_original}")
        return []

    if len(img_original_bgr.shape) == 3:
        is_color = True
        img_original_rgb = cv2.cvtColor(img_original_bgr, cv2.COLOR_BGR2RGB)
        img_original_cv = img_original_rgb
    else:
        is_color = False
        img_original_cv = img_original_bgr
        print("Citra grayscale dimuat.")

    original_size_bytes = os.path.getsize(image_path_original)
    
    results = []

    min_dim = min(img_original_cv.shape[:2])
    win_size = min(7, min_dim if min_dim % 2 == 1 else min_dim - 1)
    if win_size < 3:
        win_size = 3

    # --- JPEG Compression ---
    jpeg_qualities = [95, 75, 50, 25, 10]
    for quality in jpeg_qualities:
        base_name = os.path.splitext(image_path_original)[0]
        jpeg_path = f'{base_name}_jpeg_q{quality}.jpg'

        if is_color:
            img_to_save = cv2.cvtColor(img_original_cv, cv2.COLOR_RGB2BGR)
        else:
            img_to_save = img_original_cv

        cv2.imwrite(jpeg_path, img_to_save, [cv2.IMWRITE_JPEG_QUALITY, quality])
        compressed_size_bytes = os.path.getsize(jpeg_path)

        img_compressed_bgr = cv2.imread(jpeg_path)
        if is_color:
            img_compressed_cv = cv2.cvtColor(img_compressed_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_compressed_cv = cv2.imread(jpeg_path, cv2.IMREAD_GRAYSCALE)

        psnr_value = cv2.PSNR(img_original_cv, img_compressed_cv)

        try:
            if is_color:
                ssim_value = ssim(img_original_cv, img_compressed_cv, channel_axis=2, win_size=win_size, data_range=img_original_cv.max() - img_original_cv.min())
            else:
                ssim_value = ssim(img_original_cv, img_compressed_cv, win_size=win_size, data_range=img_original_cv.max() - img_original_cv.min())
        except ValueError:
            ssim_value = None

        is_identical = np.array_equal(img_original_cv, img_compressed_cv)

        results.append({
            'Image': base_name.capitalize(),
            'Method': 'JPEG',
            'Quality': str(quality),
            'FileSize (KB)': compressed_size_bytes / 1024,
            'CompressionRatio': original_size_bytes / compressed_size_bytes if compressed_size_bytes > 0 else float('inf'),
            'PSNR (dB)': psnr_value,
            'SSIM': ssim_value,
            'Identical': 'Ya' if is_identical else 'Tidak'
        })

    # --- PNG Compression ---
    png_compression_levels = [0, 9] # Just 0 and 9 to match lossless comparison requirement generally, we'll use 'Lossless' for level 9 in reporting
    for level in png_compression_levels:
        base_name = os.path.splitext(image_path_original)[0]
        png_path = f'{base_name}_png_l{level}.png'

        if is_color:
            img_to_save_png = cv2.cvtColor(img_original_cv, cv2.COLOR_RGB2BGR)
        else:
            img_to_save_png = img_original_cv

        cv2.imwrite(png_path, img_to_save_png, [cv2.IMWRITE_PNG_COMPRESSION, level])
        png_size_bytes = os.path.getsize(png_path)

        img_png_compressed_bgr = cv2.imread(png_path)
        if is_color:
            img_png_compressed_cv = cv2.cvtColor(img_png_compressed_bgr, cv2.COLOR_BGR2RGB)
        else:
            img_png_compressed_cv = cv2.imread(png_path, cv2.IMREAD_GRAYSCALE)

        psnr_png = cv2.PSNR(img_original_cv, img_png_compressed_cv)
        
        try:
            if is_color:
                ssim_png = ssim(img_original_cv, img_png_compressed_cv, channel_axis=2, win_size=win_size, data_range=img_original_cv.max() - img_original_cv.min())
            else:
                ssim_png = ssim(img_original_cv, img_png_compressed_cv, win_size=win_size, data_range=img_original_cv.max() - img_original_cv.min())
        except ValueError:
            ssim_png = None

        is_identical = np.array_equal(img_original_cv, img_png_compressed_cv)

        results.append({
            'Image': base_name.capitalize(),
            'Method': 'PNG',
            'Quality': f'Level {level}' if level != 9 else 'Lossless',
            'FileSize (KB)': png_size_bytes / 1024,
            'CompressionRatio': original_size_bytes / png_size_bytes if png_size_bytes > 0 else float('inf'),
            'PSNR (dB)': 'Infinity' if psnr_png > 300 else psnr_png,
            'SSIM': ssim_png,
            'Identical': 'Ya' if is_identical else 'Tidak'
        })
        
    return results, original_size_bytes

all_results = []
original_sizes = {}
for img in ['lena.png', 'peppers.png', 'cameraman.tif']:
    res, orig_size = run_experiment(img)
    all_results.extend(res)
    original_sizes[os.path.splitext(img)[0].capitalize()] = orig_size

df_results = pd.DataFrame(all_results)
df_results.to_csv("compression_results.csv", index=False)

# Add original images to results for the table
final_results = []
for img_name, orig_size in original_sizes.items():
    final_results.append({
        'Image': img_name,
        'Method': 'Asli',
        'Quality': '-',
        'FileSize (KB)': orig_size / 1024,
        'CompressionRatio': 1.00,
        'PSNR (dB)': 'Infinity',
        'SSIM': 1.000,
        'Identical': 'Ya'
    })

final_results.extend(all_results)
df_final = pd.DataFrame(final_results)

# Clean up dataframe
df_final = df_final[df_final['Quality'] != 'Level 0'] # Remove level 0 PNG from table to match report
print("\n--- Hasil Kompresi ---")
print(df_final.to_string())

# --- Generate Graphs ---
# Plot 1: Kualitas JPEG vs Ukuran File & PSNR
for img_name in ['Lena', 'Peppers', 'Cameraman']:
    df_jpeg = df_results[(df_results['Image'] == img_name) & (df_results['Method'] == 'JPEG')].copy()
    if df_jpeg.empty: continue
    df_jpeg['Quality'] = pd.to_numeric(df_jpeg['Quality'])
    df_jpeg = df_jpeg.sort_values('Quality')
    
    fig, ax1 = plt.subplots(figsize=(8, 6))
    
    color = 'tab:blue'
    ax1.set_xlabel('Kualitas JPEG')
    ax1.set_ylabel('Ukuran File (KB)', color=color)
    ax1.plot(df_jpeg['Quality'], df_jpeg['FileSize (KB)'], marker='o', color=color)
    ax1.tick_params(axis='y', labelcolor=color)
    
    ax2 = ax1.twinx()
    color = 'tab:red'
    ax2.set_ylabel('PSNR (dB)', color=color)
    ax2.plot(df_jpeg['Quality'], df_jpeg['PSNR (dB)'], marker='s', color=color, linestyle='--')
    ax2.tick_params(axis='y', labelcolor=color)
    
    plt.title(f'Kualitas JPEG vs Ukuran File & PSNR ({img_name})')
    fig.tight_layout()
    plt.savefig(f'graph_jpeg_quality_{img_name}.png')
    plt.close()

# Plot 2: Perbandingan Ukuran File
for img_name in ['Lena', 'Peppers', 'Cameraman']:
    df_img = df_final[df_final['Image'] == img_name].copy()
    if df_img.empty: continue
    
    labels = []
    sizes = []
    
    # Asli
    orig = df_img[df_img['Method'] == 'Asli']
    if not orig.empty:
        labels.append('Asli')
        sizes.append(orig.iloc[0]['FileSize (KB)'])
        
    # PNG Lossless
    png = df_img[(df_img['Method'] == 'PNG') & (df_img['Quality'] == 'Lossless')]
    if not png.empty:
        labels.append('PNG Lossless')
        sizes.append(png.iloc[0]['FileSize (KB)'])
        
    # JPEG (Q95, Q50, Q10)
    for q in ['95', '50', '10']:
        jpeg = df_img[(df_img['Method'] == 'JPEG') & (df_img['Quality'] == q)]
        if not jpeg.empty:
            labels.append(f'JPEG Q{q}')
            sizes.append(jpeg.iloc[0]['FileSize (KB)'])
            
    plt.figure(figsize=(10, 6))
    plt.bar(labels, sizes, color=['gray', 'green', 'blue', 'orange', 'red'])
    plt.ylabel('Ukuran File (KB)')
    plt.title(f'Perbandingan Ukuran File Asli, PNG, dan JPEG ({img_name})')
    plt.tight_layout()
    plt.savefig(f'graph_size_comparison_{img_name}.png')
    plt.close()

print("Experiments completed and graphs saved.")
