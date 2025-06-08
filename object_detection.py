import cv2 as cv
import numpy as np
#import matplotlib.pyplot as plt


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

def cross2d(vec_a, vec_b):
    return vec_a[0]*vec_b[1] - vec_a[1]*vec_b[0]

def order_corners(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = np.sum(pts, axis=1)
    d = np.diff(pts, axis=1)

    rect[0] = pts[np.argmin(s)]  # top-left - min of x+y
    rect[2] = pts[np.argmax(s)]  # bottom-right - max of x+y
    rect[1] = pts[np.argmin(d)]  # top-right - min of x-y
    rect[3] = pts[np.argmax(d)]  # bottom-left - max of x+y

    return rect

def around(value, target, thresh):
    #let thresh be a percent of error
    return target*(1 - thresh) <= value <= target*(1 + thresh)


#filter out convex hull to get closed contours for accurate area
#filter out area
#go back and then fit a square to the accurate object contour
#push the warped square to be straight - ie straighten it from a warped 3d object laying flat on 2d
#re-adjust and re-rotate
#then ocr
def cut_to_object(img, contour):
    #fit rectangle
    #filtered.append(closed)
    closed_pts = [i[0] for i in contour]

    marked = [0 for i in closed_pts]

    for ind in range(2, len(closed_pts)+2):
        i = ind%len(closed_pts)

        # POINT LABEL
        #cv.putText(img,str(closed_pts[i]),[closed_pts[i][0],closed_pts[i][1]+(30 if i%3==0 else (-30 if i%3==1 else 0))],cv.FONT_HERSHEY_PLAIN,1,(255,255,255))
        # DRAW COLORED LINES
        #cv.line(img,closed_pts[i-1],closed_pts[i],(255 if i%3==0 else 0,255 if i%3==1 else 0,255 if i%3==2 else 0),3)

        angle = getAngle(closed_pts[i-2],closed_pts[i-1],closed_pts[i])
        # ANGLE READOUT DISPLAY
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
        #cv.putText(img, str(pot_corners[i]), [pot_corners[i][0], pot_corners[i][1]+(30 if i%3==0 else (-30 if i%3==1 else 0))], cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        cv.circle(img, tuple(map(int, pot_corners[i])), 10, (0, 255, 255), -1)
        #cv.imwrite(f"corner? at {i}.png",img)

    print()
    print()

    #process to find 4 corners
    # 250606 note: this alg is largely inefficient calculating dists and a mean across n
    #              and then looping over n + (small) k<n
    #              even tho its tech O(n) its rly like 3*n and n doesnt get super duper large so yeah o well
    corner_start,corner_end = -1,-1
    corner_inds = set()
    dists = []
    for i in range(1, len(pot_corners)):
        dists.append(np.linalg.norm(np.array(pot_corners[i]) - np.array(pot_corners[i - 1])))
    dists.sort()
    avg = np.mean(dists[0:int(len(dists) * 3 / 2)])
    # im p sure statistically, the outliers are so big that if we avg the lower 3 quartiles we are still chill
    #plt.plot(dists, "ro")
    #plt.show()

    ind, offset = 0,0
    finalized_offset = False
    while ind < len(pot_corners)+offset+1:
        # i is the modded version to allow loopback of the loop and guarantee a full cycle where we hit every corner
        # ind is the actual loop, and its used when doing ind calc for accurate calcs w/o loopback issue
        # +1 required cuz the alg works by running midpt calc when u reach i+1 (cuz it then calcs curr index - 1 for midpt)
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

                finalized_offset=True
                offset=ind

            else:
                # now once you find another large dist, there will be a corner inside
                # (ie if large dist and start != -1)

                #   calc midpt - to i-1
                midpt = int(((ind-1)+corner_start)/2)%len(pot_corners)
                print(f"midpt at {midpt}")

                corner_inds.add(midpt)
                cv.circle(img, tuple(map(int,pot_corners[midpt])), 10, (255, 0, 0), -1)  # blue dot
                #cv.imwrite(f"corner added at {midpt}.png",img)

                #   set start to i
                corner_start = ind
                print(f"start again at {i}")
        ind+=1
        print()

    # given corners, flatten and push frame to these 4 corners
    # using perspective/affine transformations, https://math.stackexchange.com/questions/13404/mapping-irregular-quadrilateral-to-a-rectangle

    # find destination box
    corner_inds = list(corner_inds) #REMEMBER: ITEMS IN CORNERS ARE INDEXES OF POT_CORNERS
    corner_inds.sort()
    corners = [pot_corners[i] for i in corner_inds]
    corners = np.array(corners, dtype=np.float32)
    corners = order_corners(corners)
    if len(corners)!=4:
        print("ERROR: LESS THAN 4 CORNERS WERE DETERMINED")
        return None
    else:
        u = corners[1] - corners[0]
        v = corners[2] - corners[1]
        w = corners[3] - corners[2]
        a = corners[0] - corners[3]
        print("u:", u)
        print("v:", v)
        print("w:", w)
        print("a:", a)
        print("corners:",corners)
        #use the formula - it yields an aspect ratio length:width for the resized rectangle
        #UPDATE: FORMULA NO WORK
        #ratio = (1/2)*(cross2d(u,v)+cross2d(w,a))/(np.linalg.norm(u)+np.linalg.norm(v)+np.linalg.norm(w)+np.linalg.norm(a))**2
        #length = np.linalg.norm(u)
        #width = length/ratio
        #print(ratio)

        #compute destination coords
        length = max(np.linalg.norm(u),np.linalg.norm(w))
        width = max(np.linalg.norm(v),np.linalg.norm(a))
        dst = np.array([[0.0,0.0],[length,0.0],[length,width],[0.0,width]], dtype=np.float32)
        print(length,width)

        #get transformation
        M = cv.getPerspectiveTransform(corners, dst)
        img_resized = cv.warpPerspective(img,M,(int(length),int(width)))
        cv.imshow("transformed", img_resized)
        return img_resized

def get_main_object(img):
    edges = cv.Canny(img,50,100)
    contours, _ = cv.findContours(edges.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    pos_objs = []
    for c in contours:
        closed = cv.convexHull(c)
        closed_area = cv.contourArea(closed)
        #for now, use 10e4; problems in future may arise where its necessary to access the top 5 or so biggest contours
        if closed_area > 10e4:
            pos_objs.append(closed)

    return pos_objs

'''
im = cv.imread("upload.png")

cv.imshow("img",im)
while 1:
    if cv.waitKey(1) == ord("q"):
        break
'''