import numpy as np


# Histogram
def manual_cal_hist(image):
    hist = np.zeros(256, dtype = np.uint32)
    for r_pixel in image.flatten():  
        hist[r_pixel] += 1          
    return hist
    
# PDF
def normalization(image):
    h, w = image.shape[:2]
    return manual_cal_hist(image) / (h * w)

"""
# CDF
# def cdf(image):
#     hist = normalization(image)
#     for i in range(1, 256):
#         hist[i] = hist[i] + hist[i - 1]
#     return hist
"""
def cdf(image):
    return np.cumsum(normalization(image))

# Equalization 
def equalization(image):
    cdf_vals = cdf(image)
    transform = np.round((256 - 1) * cdf_vals).astype(np.uint8)
    return transform[image]

# Threshole
def bw(image, t):
    binary = np.zeros(image.shape, dtype=np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixels = image[i][j]
            if pixels > t: binary[i][j] = 255
            else: binary[i][j] = 0
    return binary

# Transform
def transform(image, a, b):
    val = a * image.astype(np.float32) + b
    return np.clip(val, 0, 255).astype(np.uint8)
""" 
# không dùng cách này OVERFLOW !
# def transform(image, a, b):
#     image_new = np.zeros(image.shape, dtype=np.uint8)
#     for i in range(image.shape[0]):
#         for j in range(image.shape[1]):
#             val = a * img[i, j] + b            
#             image_new[i, j] = np.clip(val, 0, 255)
#     return image_new
"""

# negative image
def negative(r):
    return 255 - r
def negative_img(image):
    img_neg = np.zeros((image.shape[0], image.shape[1]), dtype = np.uint8)
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            pixel = image[i][j]
            s = negative(pixel)
            img_neg[i][j] = s
    return img_neg

