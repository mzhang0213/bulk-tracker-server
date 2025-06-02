import pytesseract
import cv2 as cv

def fpsToMs(fps):
    return 1000/fps

cap = cv.VideoCapture(0)
cap.set(cv.CAP_PROP_EXPOSURE,-12)

if not cap.isOpened():
    print("Cannot open camera")
    exit()
while 1:
    ret, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow('gray_sample', gray)

    #blur = cv.GaussianBlur(gray,(5,5),0)

    denoise = cv.fastNlMeansDenoising(gray)

    thresh = cv.adaptiveThreshold(denoise,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    #_,otsuThresh = cv.threshold(blur,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)

    orb = cv.ORB().create()

    kp = orb.detect(gray,None)

    kp, des = orb.compute(gray, kp)

    #FIGURE A BETTER BOX DETECTOR THAN ORB

    img2 = cv.drawKeypoints(gray, kp, None, color=(0,255,0), flags=0)
    #cv.imshow("pts",img2)

    final = thresh
    print(pytesseract.image_to_string(final))
    cv.imshow('frame', final)

    wait = cv.waitKey(int(fpsToMs(120)))
    if wait == ord('q'):
        break
    elif wait == ord('s'):
        cv.imwrite("output.png", final)
        break

cap.release()
cv.destroyAllWindows()