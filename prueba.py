from PIL import Image,ImageChops
import math, operator
from functools import reduce
import cv2
from skimage.metrics import structural_similarity as ssim
from image_similarity_measures.quality_metrics import rmse, psnr

def rmsdiff(im1, im2):
    "Calculate the root-mean-square difference between two images"

    h = ImageChops.difference(im1, im2).histogram()

    # calculate rms
    return math.sqrt(reduce(operator.add,
        map(lambda h, i: h*(i**2), h, range(len(h)))
    ) / (float(im1.size[0]) * im1.size[1]))

img1 = Image.open("letrasProg.jpg")
img2 = Image.open("letrasBuenas.jpg")
#print(rmsdiff(img1.convert("RGB"), img2.convert("RGB")))


img1 = cv2.imread('letrasProg.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('letrasFalsas.jpg', cv2.IMREAD_GRAYSCALE)
print(ssim(img1,img2))

compare_mse = (cv2.resize(img1, (355, 500)), cv2.resize(img2, (355, 500)))
compare_ssim = (cv2.resize(img1, (355, 500)), cv2.resize(img2, (355, 500)))
#print(compare_mse)
#print(compare_ssim)





