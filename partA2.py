import cv2
import numpy as np
import matplotlib.pyplot as plt
from utils.moduleAdjust import *

def count_gr(image):
    t = np.median(image)
    c = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image[i][j]
            if pixel > t: c = c + 1
    return c

def mode():
    return np.argmax(manual_cal_hist(img))

iname = [[0, 2, 2, 3, 6], 
        [1, 2, 3, 3, 6],
        [2, 3, 3, 4, 7],
        [1, 2, 3, 3, 7],
        [0, 2, 4, 5, 6]]
img = np.array(iname, dtype=np.uint8)
h, w = img.shape[:2]
# way 1
# print(img.shape)
# cv2.imshow('test', img)
# cv2.waitKey()
# cv2.destroyAllWindows()
# way 2
plt.imshow(img, cmap='gray')
plt.savefig("original.png")

print(f"foreground = {count_gr(img)}")
print(f"background = {h * w - count_gr(img)}")

M = mode()
check1 = manual_cal_hist(img)
check2 = normalization(img)
check3 = bw(img, M)

# vector 1 chieu, not image 2D (grayscale: (H, W), color: (H, W, 3) for cv2.imwrite())
# print(check1.shape)
# print(check2.shape)

plt.figure()
plt.bar(range(256), check1)
plt.xlabel("Gray Level")
plt.ylabel("Frequency")
plt.title("Histogram")
plt.savefig("hist.png")

plt.figure()
plt.bar(range(256), check2)
plt.xlabel("Gray Level")
plt.ylabel("Frequency")
plt.title("Normalization Histogram")
plt.savefig("PDF.png")

cv2.imwrite('threshole.png', check3.astype(np.uint8))

