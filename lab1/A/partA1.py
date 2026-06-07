import cv2 
import numpy as np
import matplotlib.pyplot as plt
from utils.moduleAdjust import *

iname = [[0, 1, 1, 2], 
        [2, 3, 3, 4], 
        [4, 5, 6, 6], 
        [5, 6, 7, 7]]
img = np.array(iname, dtype=np.uint8)
print(img.shape)

t = np.median(img)   
check1 = manual_cal_hist(img)
check2 = normalization(img)
check3 = cdf(img)
check33 = equalization(img)
check4 = bw(img, t)

# negation
img_neg = negative_img(img)
cv2.imwrite('negative.jpg', img_neg)

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

plt.figure()
plt.plot(range(256), check3)
plt.xlabel("Gray Level")
plt.ylabel("Frequency")
plt.title("CDF")
plt.savefig("CDF.png")

# plt.figure()
# # plt.imshow(check4, cmap='gray')
# plt.imshow(check4, cmap='viridis')
# plt.title("Binary Image")
# plt.axis('off')
# plt.savefig("threshold.png")
cv2.imwrite('equilization.png', check33.astype(np.uint8))
cv2.imwrite('threshole.png', check4.astype(np.uint8))


