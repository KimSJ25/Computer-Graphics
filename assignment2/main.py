import cv2
import numpy as np
import matplotlib.pyplot as plt


def morphological_scale_image(image_path: str, se_shape_type: str, se_size: int):
    """
    그레이스케일 이미지에 대해 지정된 구조 요소(SE)를 사용하여 Dilation과 Erosion 연산을 수행하고 결과를 시각화합니다.
    Args:
        image_path (str): 이미지 파일의 경로.
        se_shape_type (str): 구조 요소(SE)의 형태. 'rect'(사각형), 'ellipse'(타원/원형), 'cross'(십자가) 중 선택.
        se_size (int): 구조 요소(SE)의 크기 (예: 3, 5, 7 ...).
    """
    # 1. 이미지 불러오기
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        print(f"오류: '{image_path}' 이미지를 찾을 수 없거나 로드할 수 없습니다.")
        return

    # 2. Structuring Element
    # OpenCV의 cv2.getStructuringElement를 사용하여 형태와 크기에 맞는 SE 매트릭스를 생성합니다.
    if se_shape_type == 'rect':
        se = cv2.getStructuringElement(cv2.MORPH_RECT, (se_size, se_size))
    elif se_shape_type == 'ellipse':
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (se_size, se_size))
    elif se_shape_type == 'cross':
        se = cv2.getStructuringElement(cv2.MORPH_CROSS, (se_size, se_size))
    else:
        print("오류: 지원하지 않는 SE 형태입니다. ('rect', 'ellipse', 'cross' 중 선택)")
        return

    # 3. 모폴로지 연산 수행 (Dilation 및 Erosion)
    dilated_img = cv2.dilate(img, se, iterations=1)
    eroded_img = cv2.erode(img, se, iterations=1)

    # 4. 결과 시각화
    plt.figure(figsize=(18, 6)) # 3개의 이미지를 나란히 표시하기 위해 가로 크기 증가

    # 원본 이미지 표시
    plt.subplot(1, 3, 1) # 1행 3열 중 첫 번째 칸
    plt.imshow(img, cmap='gray')
    plt.title(f"Original Image\nSize: {img.shape[1]}x{img.shape[0]}")
    plt.axis('off')

    # Dilation 결과 이미지 표시
    plt.subplot(1, 3, 2) # 1행 3열 중 두 번째 칸
    plt.imshow(dilated_img, cmap='gray')
    plt.title(f"Dilation (Enlarged)\nSE: {se_shape_type.upper()}, Size: {se_size}x{se_size}")
    plt.axis('off')

    # Erosion 결과 이미지 표시
    plt.subplot(1, 3, 3) # 1행 3열 중 세 번째 칸
    plt.imshow(eroded_img, cmap='gray')
    plt.title(f"Erosion (Shrunk)\nSE: {se_shape_type.upper()}, Size: {se_size}x{se_size}")
    plt.axis('off')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    image_file = 'vegi.jpeg'

    # 1. cross 형태, 크기 3x3 SE
    print("--- cross 형태, 크기 3x3 SE를 이용한 Dilation 및 Erosion ---")
    morphological_scale_image(image_file, se_shape_type='cross', se_size=3)