import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('original.png')
img2 = cv2.imread('add.png')
img3 = cv2.imread('sub.png')
img4 = cv2.imread('mul.png')

img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
img3_rgb = cv2.cvtColor(img3, cv2.COLOR_BGR2RGB)
img4_rgb = cv2.cvtColor(img4, cv2.COLOR_BGR2RGB)

# 이미지 크기 맞추기 
img2_resized = cv2.resize(img2_rgb, (img1_rgb.shape[1], img1_rgb.shape[0]))
img3_resized = cv2.resize(img3_rgb, (img1_rgb.shape[1], img1_rgb.shape[0]))
img4_resized = cv2.resize(img4_rgb, (img1_rgb.shape[1], img1_rgb.shape[0]))

# Addition
img_add = cv2.add(img1_rgb, img2_resized)

# Subtraction
img_sub = cv2.subtract(img1_rgb, img3_resized)

# Multiplication
img_mul = cv2.multiply(img1_rgb, img4_resized, scale=0.01)

# Division
img_div = cv2.divide(img1_rgb, img4_resized, scale=255)

# 각 연산에 대한 정보 (원본, 연산 이미지, 결과 이미지)
operations_data = [
    ("Addition", img2_resized, img_add),
    ("Subtraction", img3_resized, img_sub),
    ("Multiplication", img4_resized, img_mul),
    ("Division", img4_resized, img_div)
]

plt.figure(figsize=(15, 16))

for i, (op_name, operand_img, result_img) in enumerate(operations_data):
    # 1열: 원본 이미지
    plt.subplot(4, 3, i * 3 + 1)
    plt.imshow(img1_rgb)
    plt.title(f"Original (for {op_name})")
    plt.axis('off')

    # 2열: 연산에 사용된 이미지
    plt.subplot(4, 3, i * 3 + 2)
    plt.imshow(operand_img)
    plt.title(f"Operand (for {op_name})")
    plt.axis('off')

    # 3열: 연산 결과 이미지
    plt.subplot(4, 3, i * 3 + 3)
    plt.imshow(result_img)
    plt.title(f"{op_name} Result")
    plt.axis('off')

plt.tight_layout()
plt.show()