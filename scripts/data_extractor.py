import sys
import numpy as np
import cv2
import pytesseract
import argparse
import os
import re
import pymysql as mdb
from datetime import datetime
from dateutil.parser import parse
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

def extract_data(img,info,dinfo,dtype):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    #store_image(gray, "5gray_contours_tesseract", filen[2])
    text = pytesseract.image_to_string(gray)
    if(dtype==1):
        dinfo.append(text)
    else:
        info.append(text+"\n")
    return dinfo

def change_format(m,dt,dtype):
    month = ""
    if(dtype=="driversA" or dtype=="driversB"):
        return re.sub(r'(\d{4})\s(\d{1,2})\s(\d{1,2})', '\\1-\\2-\\3', dt)
    elif(dtype=="prc"):
        return re.sub(r'(\d{1,2})\s+(\d{1,2})\s+(\d{4})', '\\3-\\1-\\2', dt)
    else:
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
        pattern = re.compile("[^a-z]")
        if pattern.match(month):
            return re.sub(r'(\d{1,2})\s+(\d{1,2})\s+(\d{4})', '\\3-\\1-\\2', dt)
        else:
            return re.sub(r'(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})', '20\\3-\\1-\\2', dt)

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

def extract(image,m,k,h,w,info,dinfo,dtype):
    #downsample and apply grayscale
    rgb = pyrDown(image)
    gray = cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    #apply morphology
    kernel = getStructuringElement(cv2.MORPH_ELLIPSE, (m, m))
    morph = morphologyEx(gray, cv2.MORPH_GRADIENT, kernel)

    #noise removal
    _, bw = threshold(morph, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = getStructuringElement(cv2.MORPH_RECT, (k, k))
      
    # Find the contours
    _, contours,hierarchy = cv2.findContours(bw,cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)

    # For each contour, find the bounding rectangle and draw it
    for cnt in contours:
        x,y,rect_w,rect_h = cv2.boundingRect(cnt)
        if rect_h > h  and rect_w > w:
            temp = rgb.copy()
            contour = cv2.rectangle(temp,(x,y),(x+rect_w,y+rect_h),(0,255,0),2)
            cropped = contour[y:y+rect_h, x:x+rect_w]
            store_image(cropped, "5contours_tesseract", filen[2])
            #apply ocr for each component
            dinfo = extract_data(cropped,info,dinfo,dtype)
        if rect_h > h  and rect_w > w:
            contour = cv2.rectangle(rgb,(x,y),(x+rect_w,y+rect_h),(0,255,0),2)

    #store_image(contour, "4possible_contours_tesseract", filen[2])
    return dinfo 

def insert_into_db(ip,user,pswd,db,lname,fname,mname,dtype,dextracted):
    cursor.execute("""INSERT INTO cards VALUES (%s,%s,%s,%s,%s,%s)""",(0,lname,fname,mname,dtype,dextracted))
    
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

info = []
dinfo = []
extracted=""
fn = ""
mn = ""
ln = ""
m = " "
#crop rois per type
if args["type"] == "UMID":
    #extract info
    ln_roi = image[625:810, 465:1065]
    store_image(ln_roi, "2a_lname_roi", filen[2])
    extract(ln_roi,3,5,5,15,info,dinfo,0)

    fn_roi = image[810:995, 465:1065]
    store_image(fn_roi, "2a_fname_roi", filen[2])
    extract(fn_roi,3,5,5,15,info,dinfo,0)

    mn_roi = image[985:1185, 465:1065]
    store_image(mn_roi, "2a_mname_roi", filen[2])
    extract(mn_roi,3,5,15,15,info,dinfo,0)

    for i in info:
        extracted += i

    extracted = re.sub('[^A-Z\n]',' ',extracted)
    extracted = re.sub('\n',' ',extracted)
    
    #insert to db
    ln = re.sub('[^A-Z]',' ',ln)
    mn = re.sub('[^A-Z]',' ',mn)
    ln = re.sub('\n',' ',ln)
    mn = re.sub('\n',' ',mn)
    fn = re.sub('\n',' ',fn)
    print "Lastname: " + ln + "\nFirstname: " + fn + "\nMiddlename: " + mn
    print "ID Type: " + args["type"] + "\nValidity Date: No Expiration"
    # insert_into_db(ip,user,pswd,db,args["user"],uname[1],uname[2],uname[0],args["type"],None); 
else:
    if args["type"] == "driversA":
        n_roi = image[545:745, 25:2050]
        store_image(n_roi, "2a_name_roi", filen[2])
        extract(n_roi,12,150,35,20,info,dinfo,0)

        d_roi = image[1170:1295, 2310:2815]
        store_image(d_roi, "2b_date_roi", filen[2])
        dinfo =extract(d_roi,15,150,40,105,info,dinfo,1)
    elif args["type"] == "driversB":
        n_roi = image[645:765, 1035:2860]
        store_image(n_roi, "2a_name_roi", filen[2])
        extract(n_roi,17,150,35,20,info,dinfo,0)

        d_roi = image[1330:1475, 1720:2255]
        store_image(d_roi, "2b_date_roi", filen[2])
        dinfo =extract(d_roi,15,150,40,105,info,dinfo,1)
    elif args["type"] == "passport":
        # ln_roi = image[400:500, 945:1745]
        # store_image(ln_roi, "2a_lname_roi", filen[2])
        # extract(ln_roi,12,150,35,45,info,dinfo,0)

        # fn_roi = image[550:665, 945:1745]
        # store_image(fn_roi, "2a_fname_roi", filen[2])
        # extract(fn_roi,12,150,35,45,info,dinfo,0)

        # mn_roi = image[700:820, 945:1745]
        # store_image(mn_roi, "2a_mname_roi", filen[2])
        # extract(mn_roi,12,150,35,30,info,dinfo,0)

        # d_roi = image[1350:1470, 940:1745]
        # store_image(d_roi, "2b_date_roi", filen[2])
        # dinfo =extract(d_roi,25,150,15,35,info,dinfo,1)

        ln_roi = image[390:570, 1010:1825]
        store_image(ln_roi, "2a_lname_roi", filen[2])
        extract(ln_roi,10,150,35,20,info,dinfo,0)

        fn_roi = image[590:750, 1010:1825]
        store_image(fn_roi, "2a_fname_roi", filen[2])
        extract(fn_roi,12,150,35,25,info,dinfo,0)

        mn_roi = image[760:930, 1010:1825]
        store_image(mn_roi, "2a_mname_roi", filen[2])
        extract(mn_roi,12,150,35,25,info,dinfo,0)

        d_roi = image[1520:1615, 1010:1825]
        store_image(d_roi, "2b_date_roi", filen[2])
        dinfo =extract(d_roi,20,150,40,105,info,dinfo,1)
    elif args["type"] == "prc":
        ln_roi = image[705:865, 735:1745]
        store_image(ln_roi, "2a_lname_roi", filen[2])
        extract(ln_roi,10,150,35,20,info,dinfo,0)

        fn_roi = image[830:995, 735:1540]
        store_image(fn_roi, "2a_fname_roi", filen[2])
        extract(fn_roi,10,150,35,25,info,dinfo,0)

        mn_roi = image[920:1095, 735:1540]
        store_image(mn_roi, "2a_mname_roi", filen[2])
        extract(mn_roi,10,150,35,20,info,dinfo,0)

        d_roi = image[1275:1430, 735:1540]
        store_image(d_roi, "2b_date_roi", filen[2])
        dinfo =extract(d_roi,14,150,45,105,info,dinfo,1)

    if args["type"] == "driversA" or args["type"] == "driversB":
        ln,mn = info[-1], info[0]
        info.remove(ln)
        info.remove(mn)
        info.reverse()
        for i in info:
            fn += i
    elif args["type"] == "passport" or args["type"] == "prc":
        ln,mn = info[0], info[-1]
        info.remove(ln)
        info.remove(mn)
        info.reverse()
        for i in info:
            fn += i
    else:
        for i in info:
            extracted += i

        extracted = re.sub('[^A-Z\n]',' ',extracted)
        extracted = re.sub('\n',' ',extracted)
        print(extracted)

    ln = re.sub('[^A-Z]',' ',ln)
    ln = re.sub('\s+','',ln)
    mn = re.sub('[^A-Z]',' ',mn)
    ln = re.sub('\n',' ',ln)
    mn = re.sub('\n',' ',mn)
    fn = re.sub('\n',' ',fn)
    print "Lastname: " + ln + "\nFirstname: " + fn + "\nMiddlename: " + mn

    dextracted = " "
    month = " "
    for i in dinfo:
        dextracted += i

    if(args["type"]!="passport"):
        dextracted = re.sub('[^0-9]',' ',dextracted)
    else:
        month = re.sub('[^A-Za-z]','',dextracted)
        dextracted = re.sub('[^0-9]',' ',dextracted)

    dextracted = change_format(month,dextracted,args["type"])
    converted = re.sub('\\s+','',dextracted)

    print "ID Type: " + args["type"] + "\nValidity Date: " + converted
    insert_into_db(ip,user,pswd,db,ln,fn,mn,args["type"],dextracted);   

conn.commit()
conn.close()

cv2.waitKey()