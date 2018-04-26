import sys
import numpy as np
import cv2
import pytesseract
import argparse
import os
from PIL import Image
from os import listdir
from os import makedirs
from os import path
from cv2 import boundingRect, countNonZero, cvtColor, drawContours, findContours, getStructuringElement, imread, morphologyEx, pyrDown, rectangle, threshold

#======MINOR FUNCTIONS======
def store_image(img, foldern, filen):
    fpath = "results/"+foldern+"/"
    if not path.isdir(fpath):
        makedirs(fpath)
    cv2.imwrite(fpath+filen, img)

def extract_data(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #store_image(gray, "5gray_contours_tesseract", filen[2])
    text = pytesseract.image_to_string(gray)
    print(text)

def locate_points(pts):
    ret = np.zeros((4,2), dtype = "float32")
    sumF = pts.sum(axis = 1)
    diffF = np.diff(pts, axis = 1)
    ret[0] = pts[np.argmin(sumF)]
    ret[1] = pts[np.argmin(diffF)]
    ret[2] = pts[np.argmax(sumF)]
    ret[3] = pts[np.argmax(diffF)]
    return ret

def distance_formula(x1, x2, y1, y2):
    dist1 = np.sqrt(((x1[0] - x2[0]) ** 2) + ((x1[1] - x2[1]) ** 2))
    dist2 = np.sqrt(((y1[0] - y2[0]) ** 2) + ((y1[1] - y2[1]) ** 2))
    if int(dist2) < int(dist1):
        return int(dist1)
    else:
        return int(dist2)

#warp to fix perspective
def warp(img, pts):
    (tl, tr, br, bl) = locate_points(pts)
    maxW = distance_formula(br,bl,tr,tl)
    maxH = distance_formula(tr,br,tl,bl)
    dist = np.array([[0, 0],[maxW - 1, 0],[maxW - 1, maxH - 1],[0, maxH - 1]], dtype = "float32")
    transform = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype="float32"), dist)
    #warpped = cv2.warpPerspective(image, transform, (maxW, maxH))
    cropped = img[int(tl[1]):int(br[1]), int(tl[0]):int(br[0])]
    return cropped

#======LOCATING ID CARD=====
def possibleContour(cnt, fwidth, fheight):
    max_t = 0.95
    min_t= 0.3
    min_area = fwidth * fheight * min_t
    max_area = fwidth * fheight * max_t
    min_width = min_t * fwidth
    max_width = max_t * fwidth
    min_height = min_t * fheight
    max_height = max_t * fheight

    #bound area
    size = cv2.contourArea(cnt)
    if size < min_area: 
        return False
    if size > max_area:
        return False

    #get new width and height
    rect = cv2.minAreaRect(cnt)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    (tl,tr,br,bl) = locate_points(box)
    new_width = int(((br[0]-bl[0])+(tr[0]-tl[0]))/2)
    new_height = int(((br[1]-tr[1])+(bl[1]-tl[1]))/2)
    if new_width < min_width:
        return False
    if new_height < min_height:
        return False
    if new_width > max_width:
        return False
    if new_height > max_height:
        return False
        
    return True

#find largest rectangle other than the whole image
def find_card(img,filen):
    height = np.size(img, 0)
    width = np.size(img, 1)

    #convert to grayscale to find contours
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray,(1,1),1000)
    #store_image(blur, '1preprocess_gray', filen)

    #thresholding and contouring
    flag, thresh = cv2.threshold(blur, 115, 255, cv2.THRESH_BINARY)
    #store_image(thresh, '2preprocess_thresh', filen)
    im2, contours, hierarchy =  cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contoured = img.copy()

    #find largest rectangle
    max = None
    for cnt in contours:
        if possibleContour(cnt, width, height):
            contoured = cv2.drawContours(contoured, [cnt], -1, (0,255,0), 3)
            if max is None or cv2.contourArea(max) < cv2.contourArea(cnt):
                max = cnt

    #store_image(contoured, "3possible_contours", filen)

    #find corners of the card
    rect = cv2.minAreaRect(max)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box

#
def extract(image):
    #downsample and apply grayscale
    rgb = pyrDown(image)
    gray = cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    #store_image(gray, "6grayscale_tesseract", filen[2])

    #apply morphology
    kernel = getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    morph = morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)
    #store_image(morph, "7morphology_tesseract", filen[2])

    #noise removal
    _, bw = threshold(morph, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = getStructuringElement(cv2.MORPH_RECT, (5, 1))
    #store_image(bw, "8binarized_tesseract", filen[2])

    #connected components
    connected = morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
    mask = np.zeros(bw.shape, np.uint8)
    im2, contours, hierarchy = findContours(connected, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    store_image(connected, "3connected_components_tesseract", filen[2])

    #find contours
    for idx in range(0, len(hierarchy[0])):
        rect = x, y, rect_width, rect_height = boundingRect(contours[idx])
        # fill the contour
        mask = drawContours(mask, contours, idx, (255, 255, 255), cv2.FILLED)
        # ratio of non-zero pixels in the filled region
        r = float(countNonZero(mask)) / (rect_width * rect_height)
        if r > 0.5 and rect_height > 15 and rect_width > 35 and rect_height < rect_width:
            temp = rgb.copy()   #to avoid multiple rectangles on one component
            cnt = rectangle(temp, (x, y+rect_height), (x+rect_width, y), (0,255,0),2)
            cropped = cnt[y:y+rect_height, x:x+rect_width]
            store_image(cropped, "5contours_tesseract", filen[2])
            #apply ocr for each component
            extract_data(cropped)
        if r > 0.5 and rect_height > 15 and rect_width > 35 and rect_height < rect_width:
            cnt = rectangle(rgb, (x, y+rect_height), (x+rect_width, y), (0,255,0),2)
            
    store_image(cnt, "4possible_contours_tesseract", filen[2])
    return cnt
    
#=======MAIN PROGRAM========
#arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path")
ap.add_argument("-t", "--type", type = str,  required = True)
args = vars(ap.parse_args())

#load image and find square
image = cv2.imread(args["image"])
filen = args["image"].split("/")
print("Input image: " + filen[2])
card = find_card(image,filen[2])

#draw contours
contoured = cv2.drawContours(image.copy(), [card], -1, (0,255,0), 3)
#store_image(contoured, "4located_card", filen[2])

#warp and crop
image = warp(contoured, card)
store_image(image, "1perspective_crop", filen[2])

#crop rois per type
if args["type"] == "driversA":
    print("drivers_a")
    #crop name
    n_roi = image[420:570, 1:1625]
    store_image(n_roi, "2a_name_roi", filen[2])
    extract(n_roi)
    #crop date
    d_roi = image[965:1075, 1725:2200]
    store_image(d_roi, "2b_date_roi", filen[2])
    extract(d_roi)
elif args["type"] == "driversB":
    print("drivers_b")
    #crop name
    n_roi = image[485:655, 670:2165]
    store_image(n_roi, "2a_name_roi", filen[2])
    extract(n_roi)
    #crop date
    d_roi = image[1065:1230, 1250:1690]
    store_image(d_roi, "2b_date_roi", filen[2])
    extract(d_roi)
elif args["type"] == "passport":
    print("passport")
    #crop name
    n_roi = image[285:735, 730:1300]
    store_image(n_roi, "2a_name_roi", filen[2])
    extract(n_roi)
    #crop date
    d_roi = image[1125:1300, 720:1145]
    store_image(d_roi, "2b_date_roi", filen[2])
    extract(d_roi)
elif args["type"] == "prc":
    print("prc")
    #crop name
    n_roi = image[570:900, 515:1240]
    store_image(n_roi, "2a_name_roi", filen[2])
    extract(n_roi)
    #crop date
    d_roi = image[1040:1205, 510:1090]
    store_image(d_roi, "2b_date_roi", filen[2])
    extract(d_roi)
elif args["type"] == "up":
    print("up")
    #crop name
    n_roi = image[625:840, 1:1410]
    store_image(n_roi, "2a_name_roi", filen[2])
    extract(n_roi)
    #crop date
    d_roi = image[1250:1505, 550:890]
    store_image(d_roi, "2b_date_roi", filen[2])
    extract(d_roi)
    #crop sem
    s_roi = image[1150:1435, 225:515]
    store_image(s_roi, "2c_sem_roi", filen[2])
    extract(s_roi)
else: 
    print("no expiration date")


cv2.waitKey()
