import pytesseract
from PIL import Image
import numpy as np
import cv2 as cv

img = Image.open("sample.png")

img = np.array(img)
img = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
cv.imshow("frame",img)
while 1:
    if cv.waitKey(0)==ord("q"):
        break
print("ocr'ing...")
print(pytesseract.image_to_string(img))