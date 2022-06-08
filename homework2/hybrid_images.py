# Program which creates a hybrid between two images.
# One image is created using a Sobel operator and the other using Fourier Transform

# The command line input is "python hybrid_images.py imagePath1 imagePath2"
# suggested: python hybrid_images.py ./homework2_images/cat.jpg ./homework2_images/dog.jpg

# Changing the input parameters for getSobelHybrid() in main() and createMask() in getFourierHybrid
#   will affect the quality of the hybrid images

import numpy as np
import cv2
import sys
from os import path

# returns img1 and img2 from command line inputs
def getOriginals():
    if (path.exists(sys.argv[1]) & path.exists(sys.argv[2])):
        img1 = cv2.imread(sys.argv[1])
        img2 = cv2.imread(sys.argv[2])

    else:
        print('The command line input is "python hybrid_images.py imagePath1 imagePath2"')

    if (img1.shape != img2.shape):
        img1 = resize(img1, img2)

    return img1, img2

# resize img1 to shape of img2
def resize(img1, img2):
    rows, cols = img2.shape[0], img2.shape[1]
    resize = cv2.resize(img1, (cols, rows), interpolation=cv2.INTER_AREA)

    return resize

# converts image to single precision float values
def convertToFloat32(img):
    if img.dtype == "float32":
        return img
    else:
        img = np.float32(img)
        float32Img = img.copy()
        cv2.normalize(img, float32Img, 1.0, 0.0, cv2.NORM_MINMAX)
        return float32Img

# converts image to unsigned 8-bit values
def convertToInt8(img):
    if img.dtype == "uint8":
        return img
    else:
        uint8Img = img.copy()
        uint8Img = cv2.convertScaleAbs(img, alpha = 255)
        return uint8Img

# generates a sobel filter (X and Y) for a one channel image in float32
def getSobelEdgesFloat32(img, filter, sigma, ksize):
    float32Img = convertToFloat32(img)

    blur = cv2.GaussianBlur(float32Img, filter, sigma)

    gradX = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize)
    gradY = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize)

    gradX, gradY = np.square(gradX), np.square(gradY)
    grads = gradX + gradY
    grads = np.sqrt(grads)

    return grads

# generates a sobel hybrid between a blurred 1st image and a sobel filtered 2nd image
def getSobelHybrid(img1, img2, filter1, filter2, sigma1, sigma2, ksize):
    blur1 = cv2.GaussianBlur(img1, filter1, sigma1)

    b, g, r = cv2.split(img2)
    b, g, r = getSobelEdgesFloat32(b, filter2, sigma2, ksize), getSobelEdgesFloat32(g, filter2, sigma2, ksize), getSobelEdgesFloat32(r, filter2, sigma2, ksize),
    mergedSobel = cv2.merge([b,g,r])
    sobel2 = convertToInt8(mergedSobel)

    merge = cv2.bitwise_and(sobel2, blur1)

    return merge

# creates a mask for a given image
def createMask(img, filtType, freqCutoff1, freqCutoff2, shape):
    rows, cols = img.shape[0], img.shape[1]
    crow, ccol = rows//2, cols//2
    range = int(min(crow, ccol)*freqCutoff1)

    ### Filter Type
    # LPF
    if filtType == 0:
        mask = np.zeros((rows, cols), dtype = np.uint8)
        fill = 1
    # BPF & HPF
    else:
        mask = np.ones((rows, cols), dtype = np.uint8)
        fill = 0
    
    ### Shape
    # Circle
    if shape == 0:
        mask = cv2.circle(mask, (ccol, crow), range, fill, -1)
        # for BPF
        if filtType == 2:
            mask = cv2.circle(mask, (ccol, crow), int(min(crow, ccol)*freqCutoff2), 1, -1)
    # Rectangle
    else:
        mask[crow-range:crow+range+1, ccol-range:ccol+range+1] = fill
        # for BPF
        if filtType == 2:
            range2 = int(min(crow, ccol)*freqCutoff2)
            mask[crow-range2:crow+range2+1, ccol-range2:ccol+range2+1] = 1

    return mask

# generates fourier transform image with given mask
def getFourierFilteredImg(img, mask):
    f = np.fft.fft2(img)
    fShift = np.fft.fftshift(f)

    maskedMagF = mask * fShift

    fiShift = np.fft.ifftshift(maskedMagF)

    imgBack = np.fft.ifft2(fiShift)
    imgBack = np.real(imgBack)

    return imgBack

# generates a fourier filtered hybrid between each image in color
def getFourierHybrid(img1, img2):
    b1, g1, r1 = cv2.split(img1)
    b2, g2, r2 = cv2.split(img2)

    img1Chan = [b1,g1,r1]
    img2Chan = [b2,g2,r2]
    img1Filt = []
    img2Filt = []

    lpf = createMask(img1, 0, .4, 0, 1)
    visibleLPF = lpf * 255 # used for creating mask jpgs
    
    hpf = createMask(img2, 1, .1, 0, 1)
    visibleHPF = hpf * 255 # used for creating mask jpgs
    
    for channel in img1Chan:
        img1Filt.append(getFourierFilteredImg(channel, lpf))

    for channel in img2Chan:
        img2Filt.append(getFourierFilteredImg(channel, hpf))

    img1Filt = cv2.merge(img1Filt)
    img2Filt = cv2.merge(img2Filt)

    img1Filt = cv2.convertScaleAbs(img1Filt, beta = 128)
    img2Filt = cv2.convertScaleAbs(img2Filt, beta = 100)

    fourierHybrid = cv2.bitwise_and(img1Filt, img2Filt)

    return fourierHybrid

def main():
    img1, img2 = getOriginals()
    sobelHybrid = getSobelHybrid(img1, img2, (1,1), (5,5), 1, 3, 9)
    fourierHybrid = getFourierHybrid(img1, img2)

    cv2.imshow('Image 1', img1)
    cv2.imshow('Image 2', img2)
    cv2.imshow('Sobel Hybrid', sobelHybrid)
    cv2.imshow('Fourier Hybrid', fourierHybrid)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()