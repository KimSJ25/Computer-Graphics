import cv2
import numpy as np
import matplotlib.pyplot as plt

def get_structuring_element(se_shape_type: str, se_size: int):
    if se_shape_type == 'rect':
        return cv2.getStructuringElement(cv2.MORPH_RECT, (se_size, se_size))
    elif se_shape_type == 'ellipse':
        return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
    elif se_shape_type == 'cross':
        return cv2.getStructuringElement(cv2.MORPH_CROSS, (se_size, se_size))
    else:
        print("오류: 지원하지 않는 SE 형태입니다. ('rect', 'ellipse', 'cross' 중 선택)")
        return None

def apply_dilation_erosion(image_path: str, se_shape_type: str, se_size: int):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"오류: '{image_path}' 이미지를 찾을 수 없거나 로드할 수 없습니다.")
        return

    se = get_structuring_element(se_shape_type, se_size)
    if se is None:
        return

    # Dilation 및 Erosion 수행
    dilated_img = cv2.dilate(img, se, iterations=1)
    eroded_img = cv2.erode(img, se, iterations=1)

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    plt.imshow(img, cmap='gray')
    plt.title(f"Original Image\nSize: {img.shape[1]}x{img.shape[0]}")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(dilated_img, cmap='gray')
    plt.title(f"Dilation (Enlarged)\nSE: {se_shape_type.upper()}, Size: {se_size}x{se_size}")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(eroded_img, cmap='gray')
    plt.title(f"Erosion (Shrunk)\nSE: {se_shape_type.upper()}, Size: {se_size}x{se_size}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

def apply_opening_closing(image_path: str, se_shape_type: str, se_size: int):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"오류: '{image_path}' 이미지를 찾을 수 없거나 로드할 수 없습니다.")
        return

    se = get_structuring_element(se_shape_type, se_size)
    if se is None:
        return

    # Opening 및 Closing 수행
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, se)
    closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, se)

    plt.figure(figsize=(12, 10))

    # Opening 결과 시각화
    plt.subplot(2, 2, 1)
    plt.title("Original (for Opening)")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.title(f"After Opening (Noise Removed)\nSE: {se_shape_type.upper()}, Size: {se_size}x{se_size}")
    plt.imshow(opening, cmap='gray')
    plt.axis('off')

    # Closing 결과 시각화
    plt.subplot(2, 2, 3)
    plt.title("Original (for Closing)")
    plt.imshow(img, cmap='gray')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.title(f"After Closing (Holes Filled)\nSE: {se_shape_type.upper()}, Size: {se_size}x{se_size}")
    plt.imshow(closing, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # 1. Dilation & Erosion 실행
    image_file_1 = 'vegi.jpeg'
    print(f"--- {image_file_1} 이미지에 Dilation & Erosion 적용 ---")
    apply_dilation_erosion(image_file_1, se_shape_type='cross', se_size=3)

    # 2. Opening & Closing 실행
    image_file_2 = 'test.png'
    print(f"--- {image_file_2} 이미지에 Opening & Closing 적용 ---")
    apply_opening_closing(image_file_2, se_shape_type='rect', se_size=19)