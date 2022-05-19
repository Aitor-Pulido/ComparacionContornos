import numpy as np
import cv2
from PIL import Image
'''
im = cv2.imread("C:\ProyectOCR\Data\letrasCreadas.jpg")
#im = cv2.imread('C:\\Users\\Carlos\\Desktop\\Nueva carpeta\\Pruebas malas\\P6_1.jpg')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 50, 255, cv2.cv2.THRESH_TOZERO_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im, contours, -1, (0,255,0), 3)

cv2.imwrite("a2.jpg",im)


im = cv2.imread("C:\ProyectOCR\Data\letras.jpg")
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 50, 255, cv2.cv2.THRESH_TOZERO_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im, contours, -1, (0,255,0), 3)

cv2.imwrite("a.jpg",im)



im = cv2.imread('C:\\Users\\Carlos\\Desktop\\Nueva carpeta\\Pruebas malas\\letrasCreadas.jpg')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 50, 255, cv2.cv2.THRESH_TOZERO_INV)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(im, contours, -1, (0,255,0), 3)

cv2.imwrite("a3.jpg",im)

'''


im = cv2.imread('C:\\Users\\Carlos\\Desktop\\Nueva carpeta\\Pruebas malas\\letrasCreadas.jpg')
im = cv2.imread("C:\ProyectOCR\Data\letrasCreadas.jpg")

imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 50, 255, cv2.cv2.THRESH_TOZERO)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for i, cnt in enumerate(contours):
    # if the contour has no other contours inside of it
    if hierarchy[0][i][3] != -1:  # basically look for holes
        # if the size of the contour is less than a threshold (noise)
        if cv2.contourArea(cnt) < 20:
            # Fill the holes in the original image
            cv2.drawContours(im, [cnt], 0, (0, 255, 0), -1)
            #area = cv2.contourArea(cnt)
            #print(area)


cv2.imwrite("Image.jpg", im)
mask1 = cv2.inRange(im, (0, 0, 0), (10, 10,10))
cv2.imshow('mask red color',mask1)


'''
im = cv2.imread('C:\\Users\\Carlos\\Desktop\\Nueva carpeta\\Pruebas malas\\letrasCreadas.jpg')
im = cv2.imread("C:\ProyectOCR\Data\letrasCreadas.jpg")

imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 50, 255, cv2.cv2.THRESH_TOZERO)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

for i, cnt in enumerate(contours):
    # if the contour has no other contours inside of it
    if hierarchy[0][i][3] != -1:  # basically look for holes
        # if the size of the contour is less than a threshold (noise)
        if cv2.contourArea(cnt) < 90:
            # Fill the holes in the original image
            cv2.drawContours(im, [cnt], 0, (255, 255, 255), thickness=1)
            #area = cv2.contourArea(cnt)
            #print(area)


cv2.imwrite("Image2.jpg", im)
'''
















'''
v = cv2.imread('C:\\Users\\Carlos\\Desktop\\Nueva carpeta\\Pruebas malas\\letrasCreadas.jpg')
s = cv2.cvtColor(v, cv2.COLOR_BGR2GRAY)
s = cv2.Laplacian(s, cv2.CV_16S, ksize=3)
s = cv2.convertScaleAbs(s)
cv2.imwrite('nier.jpg', s)

ret, binary = cv2.threshold(s,40,255,cv2.THRESH_BINARY)
contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    if w>5 and h>10:
        cv2.rectangle(v,(x,y),(x+w,y+h),(155,155,0),1)
cv2.imwrite('nier2.jpg',v)
'''




'''
im = cv2.imread('C:\\Users\\Carlos\\Desktop\\Nueva carpeta\\Pruebas malas\\letrasCreadas.jpg')
blur = cv2.GaussianBlur(im, (15, 15), 2)
hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
lower_gray = np.array([1, 1, 1])
upper_gray = np.array([102, 102, 102])
mask = cv2.inRange(hsv, lower_gray, upper_gray)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opened_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
masked_img = cv2.bitwise_and(im, im, mask=opened_mask)
coloured = masked_img.copy()
coloured[mask == 0] = (255, 255, 255)

gray = cv2.cvtColor(coloured, cv2.COLOR_BGR2GRAY)

des = cv2.bitwise_not(gray)

contour, hier = cv2.findContours(des, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

for cnt in contour:
    cv2.drawContours(des, [cnt], 0, (0,255,0), -1)

cv2.imwrite("a4.jpg",des)
'''


'''
img = cv2.imread("C:\ProyectOCR\Data\letrasCreadas.jpg")

gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
_,binary = cv2.threshold(gray,100,255,cv2.THRESH_BINARY)
contours,hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
drawing = np.zeros((gray.shape[0], gray.shape[1], 3), dtype=np.uint8)
CountersImg = cv2.drawContours(drawing,contours, -1, (255,255,0),3)
cv2.imwrite('Contours.jpg',CountersImg)
'''


def show_image(image):
    cv2.imwrite("b2.jpg", im)
    c = cv2.waitKey()
    if c >= 0 : return -1
    return 0
image = cv2.imread('C:\\Users\\Carlos\\Desktop\\Nueva carpeta\\Pruebas malas\\P9.jpg')
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, im = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY_INV)
contours, hierarchy  = cv2.findContours(im, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
img = cv2.drawContours(image, contours, -1, (0,255,75), 2)
show_image(image)





