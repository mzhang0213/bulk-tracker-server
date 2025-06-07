from dis import distb
from fileinput import close
from tkinter.tix import INTEGER

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

#img = cv.imread("final_frame.png")
img = cv.imread("upload.png")

edges = cv.Canny(img,50,100)

contours, _ = cv.findContours(edges.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

#cv.drawContours(img,contours,-1,(0,0,255),2)

def getAngle(p1, p2, p3):
    #get vectors
    a = np.array(p1) - np.array(p2)
    b = np.array(p3) - np.array(p2)
    a_mag = (a[0]**2+a[1]**2)**(1/2)
    b_mag = (b[0]**2+b[1]**2)**(1/2)
    dot = np.dot(a,b)
    # using formula: cos(theta) = (u dot v) / |u||v|
    cosine = dot/(a_mag*b_mag + 1e-8) #1e-8 used to balance div by 0
    angle = np.arccos(np.clip(cosine,-1.0,1.0)) #sometimes result is outside of [-1,1], so adjust it and feed it to arccos
    deg = angle/np.pi*180 #convert to degrees
    return deg

def around(value, target, thresh):
    #let thresh be a percent of error
    return target*(1 - thresh) <= value <= target*(1 + thresh)


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

        marked = [0 for i in closed_pts]

        for ind in range(2, len(closed_pts)+2):
            i = ind%len(closed_pts)

            #cv.putText(img,str(closed_pts[i]),[closed_pts[i][0],closed_pts[i][1]+(30 if i%3==0 else (-30 if i%3==1 else 0))],cv.FONT_HERSHEY_PLAIN,1,(255,255,255))
            cv.line(img,closed_pts[i-1],closed_pts[i],(255 if i%3==0 else 0,255 if i%3==1 else 0,255 if i%3==2 else 0),3)

            angle = getAngle(closed_pts[i-2],closed_pts[i-1],closed_pts[i])
            #cv.putText(img,str(angle)[:6],[closed_pts[i][0],closed_pts[i][1]+(30 if i%3==0 else (-30 if i%3==1 else 0))],cv.FONT_HERSHEY_PLAIN,1,(255,255,255))
            #cv.imwrite(f"segment at {i}.png",img)

            print("x  y  angle")
            print(str(closed_pts[i][0]) + " " +
                  str(closed_pts[i][1]) + " " +
                  str(angle))

            if abs(angle-180)<10:
                print(f"marking {closed_pts[i]} and {closed_pts[i-1]} and {closed_pts[i-2]}")
                #mark the point center to the angle
                marked[i-1]=1

        pot_corners = []
        for i in range(len(marked)):
            if marked[i] == 0:
                # points marked are in the middle of line segs
                # all points not marked are parts of corners
                # thus gather points NOT MARKED
                pot_corners.append(closed_pts[i])

        for i in range(len(pot_corners)):
            cv.putText(img, str(pot_corners[i]), [pot_corners[i][0], pot_corners[i][1]+(30 if i%3==0 else (-30 if i%3==1 else 0))], cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
            cv.circle(img, tuple(map(int, pot_corners[i])), 10, (0, 255, 255), -1)
            #cv.imwrite(f"corner? at {i}.png",img)

        print()
        print()

        #process to find 4 corners
        # 250606 note: this alg is largely inefficient calculating dists and a mean across n
        #              and then looping over 2n again
        #              even tho its tech O(n) its rly like 4*n and n doesnt get super duper large so yeah o well
        corner_start,corner_end = -1,-1
        corners = set()
        dists = []
        for i in range(1, len(pot_corners)):
            dists.append(np.linalg.norm(np.array(pot_corners[i]) - np.array(pot_corners[i - 1])))
        dists.sort()
        avg = np.mean(dists[0:int(len(dists) * 3 / 2)])
        # im p sure statistically, the outliers are so big that if we avg the lower 3 quartiles we are still chill
        #plt.plot(dists, "ro")
        #plt.show()

        for ind in range(1, len(pot_corners) * 2):
            # i is the modded version to allow loopback of the loop and guarantee a full cycle where we hit every corner
            # ind is the actual loop, and its used when doing ind calc for accurate calcs w/o loopback issue
            i = ind%len(pot_corners)

            #get the vector betw pts, then get their magnitude with linalg.norm
            curr_dist = np.linalg.norm(np.array(pot_corners[i]) - np.array(pot_corners[i - 1]))
            print(
f'''point {pot_corners[i]}
dist {curr_dist}''')

            #remove outlier distances to figure out the threshold
            # WRONG: when we encounter small dist, its like 0.8 - 0.5 (curr_dist - lower_avg) --> less than 5 fo sho
            # if the curr dist is small, then minus avg will guarantee negative
                #means that 10 is lowk arbitrary and just needs to check for negative
            # thus if its >5, then its a large dist
            if curr_dist - avg > 10:
                if corner_start==-1:
                    # if start == -1, then establish st point on encountering a large dist
                    corner_start=ind
                    print(f"starting at {corner_start}")
                else:
                    # now once you find another large dist, there will be a corner inside
                    # (ie if large dist and start != -1)

                    #   calc midpt - to i-1
                    midpt = int(((ind-1)+corner_start)/2)%len(pot_corners)
                    print(f"midpt at {midpt}")

                    corners.add(midpt)

                    #   set start to i
                    corner_start = ind
                    print(f"start again at {i}")
            print()

        for i in corners:
            cv.circle(img, tuple(map(int,pot_corners[i])), 10, (255, 0, 0), -1)  # blue dot

        print(np.median(dists))
cv.drawContours(img, filtered, -1, (0,255,0), 2)
#cv.imshow("edges",edges)
cv.imshow("img",img)
while 1:
    if cv.waitKey(1) == ord("q"):
        break