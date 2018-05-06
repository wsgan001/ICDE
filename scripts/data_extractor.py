import sys
import numpy as np
import cv2
import pytesseract
import argparse
import os
import re
import pymysql as mdb
import pandas as pd
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

def change_format(m,dt,dtype):
    month = ""
    if(dtype=="driversB"):
        return re.sub(r'(\d{4})\s+(\d{1,2})\s+(\d{1,2})', '\\1-\\2-\\3', dt)
    elif(dtype=="prc"):
        return re.sub(r'(\d{1,2})\s+(\d{1,2})\s+(\d{4})', '\\3-\\1-\\2', dt)
    elif(dtype=="passport"):
        if(m=="Jan" or m=="JAN"):
            month = m
            m="01"
        elif(m=="Feb" or m=="FEB"):
            month = m
            m="02"
        elif(m=="Mar" or m=="MAR"):
            month = m
            m="03"
        elif(m=="Apr" or m=="APR"):
            month = m
            m="04"
        elif(m=="May" or m=="MAY"):
            month = m
            m="05"
        elif(m=="Jun" or m=="JUN"):
            month = m
            m="06"
        elif(m=="Jul" or m=="JUL"):
            month = m
            m="07"
        elif(m=="Aug" or m=="AUG"):
            month = m
            m="08"
        elif(m=="Sep" or m=="SEP"):
            month = m
            m="09"
        elif(m=="Oct" or m=="OCT"):
            month = m
            m="10"
        elif(m=="Nov" or m=="NOV"):
            month = m
            m="11"
        elif(m=="Dec" or m=="DEC"):
            month = m
            m="12"

        dt=m+" "+dt
        pattern = re.compile("[A-Z]")
        if pattern.match(month[-1]):
            return re.sub(r'(\d{1,2})\s+(\d{1,2})\s+(\d{4})', '\\3-\\1-\\2', dt)
        else:
            return re.sub(r'(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})', '20\\3-\\1-\\2', dt)

#======LOCATING ID CARD=====
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
    cropped = img[int(tl[1]):int(br[1]), int(tl[0]):int(br[0])]
    
    resized = cv2.resize(cropped,(int(3000),int(2000)))
    return resized

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

    #thresholding and contouring
    flag, thresh = cv2.threshold(blur, 115, 255, cv2.THRESH_BINARY)
    im2, contours, hierarchy =  cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    contoured = img.copy()

    #find largest rectangle
    max = None
    for cnt in contours:
        if possibleContour(cnt, width, height):
            contoured = cv2.drawContours(contoured, [cnt], -1, (0,255,0), 3)
            if max is None or cv2.contourArea(max) < cv2.contourArea(cnt):
                max = cnt

    #find corners of the card
    rect = cv2.minAreaRect(max)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box

#========EXTRACT INFO=======
def extract_data(img, extracted_name, extract_date, dtype):
    img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    text = pytesseract.image_to_string(img)
    text = re.sub('\s+','',text)
    if(dtype == 1): #date
        extracted_date.append(text)
    else: #name
        extracted_name.append(text+"\n")

def extract(image, me, xmin, xmax, ymin, ymax, hmin, wmin, extracted_name, extract_date, idtype, dtype):
    #downsample and apply grayscale
    rgb = pyrDown(image)
    gray = cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    #apply morphology
    if(idtype == "prc" or idtype == "passport"):
        if (idtype == "prc" and dtype == 0) or (idtype == "passport"):  kernel = getStructuringElement(cv2.MORPH_ELLIPSE, (me, me-15))
        else:   kernel = getStructuringElement(cv2.MORPH_ELLIPSE, (me, me-10))
    else:
        kernel = getStructuringElement(cv2.MORPH_ELLIPSE, (me, me))
    morph = morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

    #noise removal
    _, bw = threshold(morph, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = getStructuringElement(cv2.MORPH_RECT, (150, 150))

    # Find the contours
    _, contours,hierarchy = cv2.findContours(bw,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, find the bounding rectangle and draw it
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        if(xmin < x and x < xmax) and (ymin < y and y < ymax) and (hmin < h) and (wmin < w):
            contour = cv2.rectangle(rgb,(x,y),(x+w,y+h),(0,255,0),2)
            grayc = cv2.rectangle(gray,(x,y),(x+w,y+h),(0,255,0),2)
            cropped = grayc[y:y+h, x:x+w]
            extract_date = extract_data(cropped,extracted_name,extracted_date,dtype)
            if dtype == 0:   store_image(contour, "2a_name", filen[2])
            else:   store_image(contour, "2b_date", filen[2])
    return extracted_date
#=======MAIN PROGRAM========
#arguments
ip="localhost"
user="root"
pswd="Nutella4898"
db="icdedb"
conn = mdb.connect(ip,user,pswd,db)
cursor = conn.cursor()

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required = True, help = "path")
ap.add_argument("-t", "--type", type = str,  required = True)
args = vars(ap.parse_args())

#load image and find square
image = cv2.imread(args["image"])
filen = args["image"].split("/")
card = find_card(image,filen[2])

#draw contours
contoured = cv2.drawContours(image.copy(), [card], -1, (0,255,0), 3)

#crop and resize
image = warp(contoured, card)
store_image(image, "1perspective_crop", filen[2])

extracted_name = []
extracted_date = []
extracted = ""
dextracted = ""
month = ""
ln = ""
mn = ""
fn = ""

if(args["type"]=="invalid"):
    print "Invalid ID"
elif(args["type"]=="UMID"):
    extract(image, 15, 100, 600, 300, 600, 20, 30, extracted_name, extracted_date, args["type"], 0)
else:
    if(args["type"]=="driversA"):
        extract(image, 8, 10, 1000, 250, 350, 40, 50, extracted_name, extracted_date, args["type"], 0)
        extracted_date = extract(image, 10, 1100, 1200, 550, 650, 30, 150, extracted_name, extracted_date, args["type"], 1)
    elif(args["type"]=="driversB"):
        extract(image, 12, 300, 1500, 250, 350, 40, 50, extracted_name, extracted_date, args["type"], 0)
        extracted_date = extract(image, 18, 800, 1000, 520, 650, 40, 50, extracted_name, extracted_date, args["type"], 1)
    elif(args["type"]=="prc"):
        extract(image, 25, 300, 400, 300, 500, 40, 100, extracted_name, extracted_date, args["type"], 0)
        extracted_date = extract(image, 15, 250, 450, 620, 700, 50, 50, extracted_name, extracted_date, args["type"], 1)
    elif(args["type"]=="passport"):
        extract(image, 20, 300, 1000, 150, 400, 30, 50, extracted_name, extracted_date, args["type"], 0)
        extracted_date = extract(image, 25, 300, 500, 650, 800, 30, 50, extracted_name, extracted_date, args["type"], 1)
     

if(args["type"]=="driversB"):   ln,mn = extracted_name[0], extracted_name[-1]
else:   ln,mn = extracted_name[-1], extracted_name[0]

extracted_name.remove(ln)
extracted_name.remove(mn)
if(args["type"]!="UMID" and args["type"]!="driversB"):   extracted_name.reverse()
for i in extracted_name:
    fn += i
extracted = re.sub('[^A-Z]',' ',extracted)
extracted = re.sub('\n',' ',extracted)
ln = re.sub('[^A-Z]',' ',ln)
mn = re.sub('[^A-Z]',' ',mn)
ln = re.sub('\\s+','',ln)
ln = re.sub('\n',' ',ln)
mn = re.sub('\n',' ',mn)
fn = re.sub('\n',' ',fn)

if(args["type"]=="UMID"):
    cursor.execute("""INSERT INTO cards VALUES (%s,%s,%s,%s,%s,%s)""",(0,ln,fn,mn,args["type"],None))
    print "Lastname: " + ln + "\nFirstname: " + fn + "\nMiddlename: " + mn + "\nID type: " + args["type"] + "\nValid until: No expiration" 
else:
    for i in extracted_date:
        dextracted += i
    if(args["type"]=="passport"):
        month = re.sub('[^A-Za-z]','',dextracted)
        dextracted = re.sub('[^0-9]',' ',dextracted)
    elif(args["type"]!="driversA"):
        dextracted = re.sub('[^0-9]',' ',dextracted)
    if(args["type"]=="driversA"):
        cursor.execute("""INSERT INTO cards VALUES (%s,%s,%s,%s,%s,%s)""",(0,ln,fn,mn,args["type"],dextracted))
        print "Lastname: " + ln + "\nFirstname: " + fn + "\nMiddlename: " + mn + "\nID type: " + args["type"] + "\nValid until: " + dextracted
    else:
        dextracted = change_format(month,dextracted,args["type"])
        converted = re.sub('\\s+','',dextracted)
        cursor.execute("""INSERT INTO cards VALUES (%s,%s,%s,%s,%s,%s)""",(0,ln,fn,mn,args["type"],converted))
        print "Lastname: " + ln + "\nFirstname: " + fn + "\nMiddlename: " + mn + "\nID type: " + args["type"] + "\nValid until: " + converted
    
conn.commit()
conn.close()

cv2.waitKey()