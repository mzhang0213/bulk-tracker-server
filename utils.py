import os
from datetime import datetime
import eventlet
import logging

eventlet.monkey_patch()
logging.basicConfig(level=logging.DEBUG)

import cv2 as cv
import pytesseract
import matplotlib.pyplot as plt


UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')

def saveimg_cv(scan_type, img):
    rn = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"scan-{scan_type}-{rn}.png"
    cv.imwrite(str(os.path.join(UPLOAD_FOLDER,filename)), img)
    return filename
def plot_img(img, saveImage):
    fig,ax = plt.subplots()
    plt.axis('off')
    rgb_img = cv.cvtColor(img,cv.COLOR_BGR2RGB)
    ax.imshow(rgb_img)
    if saveImage!="":
        plt.savefig('output_image.png', bbox_inches='tight', pad_inches=0)
    plt.show()
    return None

def uprint(msg):
    logging.info(msg)

def udisp_img(img, title):
    obj = saveimg_cv(title,img)
    if obj:
        return {
            "name":obj,
            "scan_type":title
        }