import numpy as np
import cv2
from PIL import Image

im = cv2.imread("C:\ProyectOCR\Data\letrasCreadas.jpg")
#im = cv2.imread("C:\ProyectOCR\Data\letras.jpg")

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
cv2.imwrite("Image2.jpg", mask1)



im = cv2.imread('C:\\Users\\Carlos\\Desktop\\Nueva carpeta\\Pruebas malas\\letrasCreadas.jpg')
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
cv2.imwrite("Image3.jpg", mask1)


# Para invertir el color
im = cv2.imread("C:\ProyectOCR\Data\letras.jpg")
invertida = cv2.bitwise_not(im)
cv2.imwrite("letrasMalas.jpg", invertida)

image = Image.open("letrasMalas.jpg")
image.thumbnail((400, 400))
image.save('letrasProg.jpg')

image = Image.open("Image2.jpg")
image.thumbnail((400, 400))
image.save('letrasBuenas.jpg')

image = Image.open("Image3.jpg")
image.thumbnail((400, 400))
image.save('letrasFalsas.jpg')

image = cv2.imread("letrasProg.jpg", 0)
number_of_white_pix = np.sum(image >= 1)
number_of_black_pix = np.sum(image == 0)
print("Letras programa")
print('Number of white pixels:', number_of_white_pix)
print('Number of black pixels:', number_of_black_pix)


image = cv2.imread("letrasBuenas.jpg", 0)
number_of_white_pix = np.sum(image >= 1)
number_of_black_pix = np.sum(image == 0)
print("Letras de verdad")
print('Number of white pixels:', number_of_white_pix)
print('Number of black pixels:', number_of_black_pix)


image = cv2.imread("letrasBuenas.jpg", 0)
number_of_white_pix = np.sum(image >= 1)
number_of_black_pix = np.sum(image == 0)
print("Letras falsas")
print('Number of white pixels:', number_of_white_pix)
print('Number of black pixels:', number_of_black_pix)



'''
image = cv2.imread("letrasFalsas.jpg")
mask1 = cv2.inRange(image, (0, 0, 0), (255, 255,255))
target = cv2.bitwise_and(image,image, mask=mask1)
cv2.imshow('target colors extracted',target)
cv2.waitKey()
'''


