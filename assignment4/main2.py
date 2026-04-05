import cv2
import numpy as np
import matplotlib.pyplot as plt

# test.png 이미지 불러오기 (grayscale로 읽음)
img = cv2.imread('test.png', cv2.IMREAD_GRAYSCALE)

if img is None:
    print("test.png 이미지를 찾을 수 없습니다. 경로를 확인하거나 파일을 main.py와 동일한 폴더에 넣어주세요.")
    exit() # 이미지를 찾지 못하면 프로그램 종료

height, width = img.shape

# SE 정의
# 19x19 크기의 사각형
se = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 19))

# Opening: Erosion -> Dilation 
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, se)

# Closing: Dilation -> Erosion 
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se)

plt.figure(figsize=(12, 10))

# Opening 결과
plt.subplot(2, 2, 1)
plt.title("Original (with White Noise)")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("After Opening (Noise Removed)")
plt.imshow(opening, cmap='gray')
plt.axis('off')

# Closing 결과
plt.subplot(2, 2, 3)
plt.title("Original (with Black Holes)")
plt.imshow(img, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("After Closing (Holes Filled)")
plt.imshow(closing, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()