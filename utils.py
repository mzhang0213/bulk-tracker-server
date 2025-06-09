import os
from datetime import datetime
import eventlet
import logging

eventlet.monkey_patch()
logging.basicConfig(level=logging.DEBUG)

import cv2 as cv
import pytesseract


UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')

def saveimg_cv(scan_type, img):
    rn = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scan-{scan_type}-{rn}.png"
    cv.imwrite(os.path.join(UPLOAD_FOLDER,filename), img)
    return {
        "name":filename,
        "scan_type":scan_type
    }

def scan_text(img):
    t1 = cv.getTickCount()

    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    #blur = cv.GaussianBlur(gray,(5,5),0)

    denoise = cv.fastNlMeansDenoising(gray)

    thresh = cv.adaptiveThreshold(denoise,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    #_,otsuThresh = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    final = thresh
    final_file = saveimg_cv("processed", final)

    logging.info("finished")
    logging.info("elapsed: "+str((cv.getTickCount()-t1)/cv.getTickFrequency()))

    return final_file,pytesseract.image_to_string(final)
