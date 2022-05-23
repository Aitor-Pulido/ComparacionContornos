import cv2
import numpy as np
from PIL import Image, ImageChops, ImageStat



'''
#No hay suficientes diferencias
height = 312
width = 312
A = cv2.imread("letrasProg.jpg", 0)

B = cv2.imread("letrasBuenas.jpg", 0)

errorL2 = cv2.norm( A, B, cv2.NORM_L2 )
similarity = 1 - errorL2 / ( height * width )
print('Similarity = ',similarity)

cv2.imshow('A',A)
cv2.imshow('B',B)
cv2.waitKey(0)
'''


'''
#Siempre me da la maxima puntuacion

base = cv2.imread('letrasProg.jpg')
test = cv2.imread('letrasFalsas.jpg')

hsv_base = cv2.cvtColor(base, cv2.COLOR_BGR2HSV)
hsv_test = cv2.cvtColor(test, cv2.COLOR_BGR2HSV)

h_bins = 50
s_bins = 60
histSize = [h_bins, s_bins]
h_ranges = [0, 180]
s_ranges = [0, 256]
ranges = h_ranges + s_ranges
channels = [0, 1]

hist_base = cv2.calcHist([hsv_base], channels, None, histSize, ranges, accumulate=False)
cv2.normalize(hist_base, hist_base, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
hist_test = cv2.calcHist([hsv_test], channels, None, histSize, ranges, accumulate=False)
cv2.normalize(hist_test, hist_test, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)

compare_method = cv2.HISTCMP_CORREL

base_base = cv2.compareHist(hist_base, hist_base, compare_method)
base_test = cv2.compareHist(hist_base, hist_test, compare_method)

print('base_base Similarity = ', base_base)
print('base_test Similarity = ', base_test)

cv2.imshow('base',base)
cv2.imshow('test1',test)
cv2.waitKey(0)
'''


'''
#Se solapan pero necesito que esten en la misma posicion
img1 = Image.open("letrasProg.jpg")
img2 = Image.open("letrasBuenas.jpg")
diff = ImageChops.difference(img1.convert("RGB"), img2.convert("RGB"))
diff.show()
'''

