import cv2
import numpy as np
import matplotlib.pyplot as plt


def morphological_scale_image(image_path: str, se_shape_type: str, se_size: int):
    """
    Dilation과 Erosion의 연산
    Args:
        se_shape_type (str): SE. 'rect', 'ellipse', 'cross'.
        se_size (int): size of SE (예: 3, 5, 7 ...).
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"오류: '{image_path}' 이미지를 찾을 수 없거나 로드할 수 없습니다.")
        return

    # Structuring Element
    if se_shape_type == 'rect':
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (se_size, se_size))
    elif se_shape_type == 'ellipse':
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
    elif se_shape_type == 'cross':
        se = cv2.getStructuringElement(cv2.MORPH_CROSS, (se_size, se_size))
    else:
        print("오류: 지원하지 않는 SE 형태입니다. ('rect', 'ellipse', 'cross' 중 선택)")
        return

    # 3. Dilation 및 Erosion
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

if __name__ == "__main__":
    image_file = 'vegi.jpeg'

    # cross , 3x3 SE
    print("--- cross 형태, 크기 3x3 SE를 이용한 Dilation 및 Erosion ---")
    morphological_scale_image(image_file, se_shape_type='cross', se_size=3)