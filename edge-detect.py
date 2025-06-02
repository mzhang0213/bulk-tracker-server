import cv2 as cv
import numpy as np

#img = cv.imread("final_frame.png")
img = cv.imread("upload.png")

edges = cv.Canny(img,50,100)

contours, _ = cv.findContours(edges.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

rectangles = []

for contour in contours:
    # 3. Approximate contour shape
    peri = 0.1*cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour, peri, True)

    # 4. Check for quadrilateral with convex shape
    if len(approx) == 4:
        area = cv.contourArea(approx)
        if area > 1:  # Minimum area threshold
            rect = cv.minAreaRect(approx)
            box = cv.boxPoints(rect)
            box = np.int32(box)
            cv.drawContours(img,[box],0,(0,0,255),2)
            rectangles.append(approx)
            # 5. Check convexity and aspect ratio
#            if cv.isContourConvex(approx):
#                rectangles.append(approx)

print(len(rectangles))
#contours_disp = cv.drawContours(img,rectangles,-1,(0,0,255),3)

cv.imshow("edges",edges)
cv.imshow("contours",img)
while 1:
    if cv.waitKey(1)==ord("q"):
        break