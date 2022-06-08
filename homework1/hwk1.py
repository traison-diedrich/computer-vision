
# hwk1.py by Traison Diedrich
# program takes in an image and creates many different versions of it
# from the command line: "python hwk1.py 'image path' 'desired block size'"
# suggested: "python hwk1.py san_francisco.jpg 100"
# when done viewing images, press any key to exit the program

import cv2
import numpy as np
import random
import sys

random.seed()

# returns original image
def getOriginal():

    img = cv2.imread (sys.argv[1])

    return img

# returns cropped image
def getCropped(img):

    blk_size = int(sys.argv[2])

    # crop image to a multiple of blk_size
    h_extra = img.shape[0] % blk_size
    w_extra = img.shape[1] % blk_size

    crop_img = img[0:(img.shape[0]-h_extra),0:(img.shape[1]-w_extra)]

    # compute num horz & vert blks
    h_blks = crop_img.shape[0] / blk_size
    w_blks = crop_img.shape[1] / blk_size

    return crop_img, h_blks, w_blks

# returns a random block from 5 color options
def getRandColorBlock(imgBlock):

    blk_size = int(sys.argv[2])
    version = random.randrange(0,5)

    # red channel
    if version == 0:
        r = imgBlock.copy()
        r[:,:,0] = 0
        r[:,:,1] = 0
        return r

    # green channel
    if version == 1:
        g = imgBlock.copy()
        g[:,:,0] = 0
        g[:,:,2] = 0
        return g
    if version == 2:
        b = imgBlock.copy()
        b[:,:,1] = 0
        b[:,:,2] = 0
        return b

    # grayscale
    if version == 3:
        gs = cv2.cvtColor(imgBlock, cv2.COLOR_BGR2GRAY)
        threeChannel = np.zeros((int(blk_size), int(blk_size), 3), dtype = np.uint8)
        threeChannel[:,:,0] = gs
        threeChannel[:,:,1] = gs
        threeChannel[:,:,2] = gs
        return threeChannel

    # reverse colors
    if version == 4:
        rev = cv2.cvtColor(imgBlock, cv2.COLOR_BGR2RGB)
        return rev     

# returns image with random color-channel versions
def getRandColImage(crop_img, h_blks, w_blks):
    randBlocks = crop_img.copy()
    blk_size = int(sys.argv[2])

    for r in range(int(h_blks)):
        for c in range(int(w_blks)):
            blk = crop_img[r*blk_size:r*blk_size+blk_size,
                    c*blk_size:c*blk_size+blk_size]
            randBlocks[blk_size*r:blk_size*r+blk_size,
                    blk_size*c:blk_size*c+blk_size] = getRandColorBlock(blk)

    return randBlocks

# returns a random gradient block of varying types and colors
def getGradientBlk():

    blkSize = int(sys.argv[2])

    # background will be either black or white
    blackWhite = random.randrange(0,2)

    if blackWhite == 0:
        blk = np.zeros((blkSize, blkSize, 3), dtype = np.uint8)

    if blackWhite == 1:
        blk = np.full((blkSize, blkSize, 3), 255, dtype = np.uint8)

    multiplier = 255/((blkSize*2)-2)

    colors = ['red', 'blue', 'green', 'light blue', 'pink', 'yellow']

    colorChoice = random.randrange(len(colors))

    channel1 = None
    channel2 = None

    # channels for black background
    if blackWhite == 0:
        if colors[colorChoice] == 'red':
            channel1 = 2
        if colors[colorChoice] == 'green':
            channel1 = 1
        if colors[colorChoice] == 'blue':
            channel1 = 0
        if colors[colorChoice] == 'light blue':
            channel1 = 0
            channel2 = 1
        if colors[colorChoice] == 'pink':
            channel1 = 0
            channel2 = 2
        if colors[colorChoice] == 'yellow':
            channel1 = 1
            channel2 = 2

    # channels for white background
    if blackWhite == 1:
        if colors[colorChoice] == 'red':
            channel1 = 0
            channel2 = 1
        if colors[colorChoice] == 'blue':
            channel1 = 1
            channel2 = 2
        if colors[colorChoice] == 'green':
            channel1 = 0
            channel2 = 2
        if colors[colorChoice] == 'light blue':
            channel1 = 2
        if colors[colorChoice] == 'pink':
            channel1 = 1
        if colors[colorChoice] == 'yellow':
            channel1 = 0
    
    # choosing which type of gradient to use
    gradStructure = random.randrange(2)
    gradType = random.randrange(4)

    # all top to bot and left to right gradients
    if gradStructure == 0:
        for r in range(blkSize):
            if gradType == 0:
                blk[r,:,channel1] = multiplier * r
                if channel2:
                    blk[r,:,channel2] = multiplier * r
            if gradType == 1:
                blk[r,:,channel1] = multiplier * (blkSize-r)
                if channel2:
                    blk[r,:,channel2] = multiplier * (blkSize-r)
            if gradType == 2:
                blk[:,r,channel1] = multiplier * r
                if channel2:
                    blk[:,r,channel2] = multiplier * r
            if gradType == 3:
                blk[:,r,channel1] = multiplier * (blkSize - r)
                if channel2:
                    blk[:,r,channel2] = multiplier * (blkSize - r)

    # all diagonal gradients
    if gradStructure == 1:
        for r in range(blkSize):
            for c in range(blkSize):
                if gradType == 0:
                    blk[r,c,channel1] = multiplier*(r+c)
                    if channel2:
                        blk[r,c,channel2] = multiplier*(r+c)
                if gradType == 1:
                    blk[r,c,channel1] = multiplier*(2*blkSize - (r+c))
                    if channel2:
                        blk[r,c,channel2] = multiplier*(2*blkSize - (r+c))
                if gradType == 2:
                    blk[r,c,channel1] = multiplier*(r+(blkSize-c))
                    if channel2:
                        blk[r,c,channel2] = multiplier*(r+(blkSize-c))
                if gradType == 3:
                    blk[r,c,channel1] = multiplier*((blkSize-r)+c)
                    if channel2:
                        blk[r,c,channel2] = multiplier*((blkSize-r)+c)

    return blk

# returns image with 1/chance of blocks being a random gradient
def getGradientImage(croppedImg, hBlks, wBlks, chance):

    blkSize = int(sys.argv[2])
    gradBlks = croppedImg.copy()

    for r in range(int(hBlks)):
        for c in range(int(wBlks)):
            isGradBlock = random.randrange(chance)
            if isGradBlock == 0:
                gradBlks[blkSize*r:blkSize*r+blkSize,
                    blkSize*c:blkSize*c+blkSize] = getGradientBlk()

    return gradBlks

# returns a version of img with randomized blocks
def getPuzzleImage(img, hBlks, wBlks):

    blkList = []
    puzzleList = []
    blkSize = int(sys.argv[2])

    puzzleImg = img.copy()

    # create list of all blocks in img
    for r in range(int(hBlks)):
        for c in range(int(wBlks)):
            blkList.append(img[blkSize*r:blkSize*r+blkSize,
                blkSize*c:blkSize*c+blkSize])

    # randomly order all blocks into puzzleList
    while len(blkList) > 0:
        piece = random.randrange(len(blkList))
        puzzleList.append(blkList.pop(piece))

    # reassemble image from puzzleList
    x = 0
    for r in range(int(hBlks)):
        for c in range(int(wBlks)):
            puzzleImg[blkSize*r:blkSize*r+blkSize,
                blkSize*c:blkSize*c+blkSize] = puzzleList[x]
            x += 1

    return puzzleImg

# returns a randomly filtered block
def getFilteredBlock(blk):

    # selecting random filter type
    filtType = random.randrange(7)

    # selecting random filter, sigma, and ksize for all filters
    filters = [(3,3), (5,5), (7,7), (9,9)]
    sigmas = [1.0, 2.0, 3.0, 4.0]
    ksizes = [1, 3, 5, 7]
    filt = filters[random.randrange(len(filters))]
    sigma = sigmas[random.randrange(len(sigmas))]
    ksize = ksizes[random.randrange(len(ksizes))]

    # box filter
    if filtType == 0:
        filtBlk = cv2.boxFilter(blk, -1, filt)
    # gaussian blur filter
    if filtType == 1:
        filtBlk = cv2.GaussianBlur(blk, filt, sigma)
    # median blur
    if filtType == 2:
        filtBlk = cv2.medianBlur(blk, ksize)
    # gaussian sharpening filter
    if filtType == 3:
        blur = cv2.GaussianBlur(blk, filt, sigma)
        filtBlk = np.multiply(blk,2) - blur
    # laplacian filter
    if filtType == 4:
        tempBlk = cv2.Laplacian(blk, -1, cv2.CV_16S, ksize)
        filtBlk = cv2.convertScaleAbs(tempBlk)
    # sobel filter x
    if filtType == 5:
        tempBlk = cv2.Sobel(blk, cv2.CV_16S, 1, 0, ksize)
        filtBlk = cv2.convertScaleAbs(tempBlk)
    # sobel filter y
    if filtType == 6:
        tempBlk = cv2.Sobel(blk, cv2.CV_16S, 0, 1, ksize)
        filtBlk = cv2.convertScaleAbs(tempBlk)

    return filtBlk

# returns a version of the image with randomly filtered blocks
def getFilteredImage(img, hBlks, wBlks):

    blkSize = int(sys.argv[2])
    filtImg = img.copy()

    for r in range(int(hBlks)):
        for c in range(int(wBlks)):
            blk = img[blkSize*r:blkSize*r+blkSize,blkSize*c:blkSize*c+blkSize]
            filtImg[blkSize*r:blkSize*r+blkSize,
                blkSize*c:blkSize*c+blkSize] = getFilteredBlock(blk)

    return filtImg

# returns an image with a combination of all previous methods
def getCombinedImage(img, hBlks, wBlks):
    puzzleImg = getPuzzleImage(img, hBlks, wBlks)
    randImg = getRandColImage(puzzleImg, hBlks, wBlks)
    gradImg = getGradientImage(randImg, hBlks, wBlks, 8)
    combImg = getFilteredImage(gradImg, hBlks, wBlks)

    return combImg

def main():

    img = getOriginal()
    cv2.imshow('1: Original Image', img)

    croppedImg, hBlks, wBlks = getCropped(img)

    randImg = getRandColImage(croppedImg, hBlks, wBlks)
    cv2.imshow('2: Random Color Blocks', randImg)

    gradImg = getGradientImage(croppedImg, hBlks, wBlks, 3)
    cv2.imshow('3: Gradient Blocks', gradImg)

    puzzleImg = getPuzzleImage(croppedImg, hBlks, wBlks)
    cv2.imshow('4: Puzzle Image', puzzleImg)

    filtImg = getFilteredImage(croppedImg, hBlks, wBlks)
    cv2.imshow('5: Filtered Blocks', filtImg)

    combImg = getCombinedImage(croppedImg, hBlks, wBlks)
    cv2.imshow('6: Combined Image', combImg)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()