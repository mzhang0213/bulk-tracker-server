from fileinput import close
from tkinter.tix import INTEGER

import cv2 as cv
import numpy as np

#img = cv.imread("final_frame.png")
img = cv.imread("upload.png")

edges = cv.Canny(img,50,100)

contours, _ = cv.findContours(edges.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

#cv.drawContours(img,contours,-1,(0,0,255),2)

#filter out convex hull to get closed contours for accurate area
#filter out area
#go back and then fit a square to the accurate object contour
#push the warped square to be straight - ie straighten it from a warped 3d object laying flat on 2d
#re-adjust and re-rotate
#then ocr
filtered = []
for contour in contours:
    closed = cv.convexHull(contour)
    closed_area = cv.contourArea(closed)
    if closed_area > 10e4:
        #fit rectangle
        #filtered.append(closed)
        closed_pts = [i[0] for i in closed]
        for i in closed_pts: print(i)
        cv.circle(img,closed_pts[0],30,(0,255,255))

        pts = contour[:, 0, :]
        x = pts[:, 0]
        y = pts[:, 0]
        dx = np.gradient(x)
        dy = np.gradient(y)
        ddx = np.gradient(dx)
        ddy = np.gradient(dy)

        eps = 1e-8
        curvature = np.abs(dx * ddy - dy * ddx) / (dx**2 + dy**2 + eps)**1.5

        # Optional: Normalize curvature and threshold
        curvature = np.nan_to_num(curvature)
        print(curvature)
        threshold = np.percentile(curvature, 95)  # top 5% curvy points
        corner_indices = np.where(curvature > threshold)[0]

        # Draw detected corners
        for i in corner_indices:
            cv.circle(img, tuple(pts[i]), 3, (255, 0, 0), -1)  # blue dot

cv.drawContours(img, filtered, -1, (0,255,0), 2)
#cv.imshow("edges",edges)
cv.imshow("img",img)
while 1:
    if cv.waitKey(1) == ord("q"):
        break