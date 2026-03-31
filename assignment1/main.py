import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import metrics

img = cv2.imread('cat.jpg', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("이미지를 찾을 수 없습니다. 경로를 확인해주세요.")
else:
    # Global HE
    he_res = cv2.equalizeHist(img)

    # AHE
    ahe_tool = cv2.createCLAHE(clipLimit=40.0, tileGridSize=(8, 8))
    ahe_res = ahe_tool.apply(img)

    # CLAHE
    clahe_tool = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_res = clahe_tool.apply(img)

    methods = [
        ('Original', img),
        ('Global HE', he_res),
        ('AHE', ahe_res),
        ('CLAHE', clahe_res)
    ]

    plt.figure(figsize=(16, 20))

    for i, (name, processed) in enumerate(methods):
        if name != 'Original':
            mse = metrics.mean_squared_error(img, processed)
            psnr = metrics.peak_signal_noise_ratio(img, processed)
            ssim = metrics.structural_similarity(img, processed)
            info_text = f"MSE: {mse:.2f}\nPSNR: {psnr:.2f}dB\nSSIM: {ssim:.4f}"
        else:
            info_text = "Reference"

        # 이미지 출력
        plt.subplot(4, 2, 2*i + 1)
        plt.imshow(processed, cmap='gray')
        plt.title(f"{name} Image")
        plt.text(10, 30, info_text, color='yellow', fontsize=12, bbox=dict(facecolor='black', alpha=0.6))
        plt.axis('off')

        # 히스토그램 출력
        plt.subplot(4, 2, 2*i + 2)
        plt.hist(processed.ravel(), 256, [0, 256], color='black')
        plt.title(f"{name} Histogram")

    plt.tight_layout()
    plt.show()