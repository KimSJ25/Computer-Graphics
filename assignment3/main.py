import cv2
import matplotlib.pyplot as plt

img1_gray = cv2.imread('1.jpg', cv2.IMREAD_GRAYSCALE)
img2_gray = cv2.imread('2.jpg', cv2.IMREAD_GRAYSCALE)

img2_gray_resized = cv2.resize(img2_gray, (img1_gray.shape[1], img1_gray.shape[0]))

# Binary image
_, mask1 = cv2.threshold(img1_gray, 127, 255, cv2.THRESH_BINARY)
_, mask2 = cv2.threshold(img2_gray_resized, 127, 255, cv2.THRESH_BINARY)

img_and = cv2.bitwise_and(mask1, mask2)
img_or = cv2.bitwise_or(mask1, mask2)
img_not = cv2.bitwise_not(mask1)

titles = ['Mask 1', 'Mask 2', 'AND Operation', 'OR Operation', 'NOT Operation']
images = [mask1, mask2, img_and, img_or, img_not]

plt.figure(figsize=(15, 5))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(images[i], cmap='gray') 
    plt.title(titles[i])
    plt.axis('off')

plt.tight_layout()
plt.show()