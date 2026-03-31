import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. 샘플 이진 이미지 생성 (검은 배경에 흰색 사각형)
img = np.zeros((300, 300), dtype=np.uint8)
cv2.rectangle(img, (50, 50), (250, 250), 255, -1)

# 잡음 추가 (Opening 테스트용: 흰색 점들 / Closing 테스트용: 검은색 구멍들)
noise_white = np.random.randint(0, 2, (300, 300), dtype=np.uint8) * 255
noise_black = np.random.randint(0, 2, (300, 300), dtype=np.uint8) * 255
img_opening_test = cv2.bitwise_or(img, noise_white) # 외부에 흰 점 추가
img_closing_test = cv2.bitwise_and(img, noise_black) # 내부에 검은 구멍 추가

# 2. 구조 요소(Structuring Element) 정의 [cite: 17, 137]
# 5x5 크기의 사각형 모양 구조 요소 사용
se = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))

# 3. Morphological Operations 수행 [cite: 238, 295]
# Opening: Erosion -> Dilation 
opening = cv2.morphologyEx(img_opening_test, cv2.MORPH_OPEN, se)

# Closing: Dilation -> Erosion 
closing = cv2.morphologyEx(img_closing_test, cv2.MORPH_CLOSE, se)

# 4. 결과 시각화 (원본과 변환 이미지를 한 화면에 출력)
plt.figure(figsize=(12, 10))

# Opening 결과 출력
plt.subplot(2, 2, 1)
plt.title("Original (with White Noise)")
plt.imshow(img_opening_test, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("After Opening (Noise Removed)")
plt.imshow(opening, cmap='gray')
plt.axis('off')

# Closing 결과 출력
plt.subplot(2, 2, 3)
plt.title("Original (with Black Holes)")
plt.imshow(img_closing_test, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("After Closing (Holes Filled)")
plt.imshow(closing, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()