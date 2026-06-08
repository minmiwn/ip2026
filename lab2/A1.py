import cv2
from matplotlib import image
import numpy as np
import matplotlib.pyplot as plt
from utils.moduleAdjust import *


# A1
image1 = [[104, 100, 108],
         [99, 106, 98],
         [95, 90, 85]]
img1 = np.array(image1, dtype=np.uint8)
plt.figure()
plt.imshow(img1, cmap='gray')
plt.savefig("a1.png")

image2 = 1/9 * img1
img2 = np.array(image2, dtype=np.uint8)
plt.figure()
plt.imshow(img2, cmap='gray')
plt.savefig("weight1.png")

image3 = 1/16 * img1
img3 = np.array(image3, dtype=np.uint8)
plt.figure()
plt.imshow(img3, cmap='gray')
plt.savefig("weight2.png")


# A2
image2 = [[12, 13, 14],
          [15, 255, 16],
          [14, 13, 12]]
img2 = np.array(image2, dtype=np.uint8)
plt.figure()
plt.imshow(img2, cmap='gray')
plt.savefig("a2.png")

# Median filter
img2_cop = img2.copy()
img2_cop[1,1] = np.mean(img2)
cv2.imshow('Median', img2_cop)
cv2.waitKey()
cv2.destroyAllWindows()

# Laplacian cho âm lớn -> trừ : pixel nổi bật/ sắc hơn
