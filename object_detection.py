import logging

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import pytesseract

from utils import uprint
from utils import udisp_img

BLUE = "\033[34m"
RESET = "\033[0m"

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


# processing of full image
def mark_corners(img,contour):
    """
    Find possible points to be corners in the contour
    :param img: image with objects being marked
    :param contour: contour from image with corners being marked
    :return: a list of potential corners, output files
    """
    output_files = []

    # resize current contour arr
    closed_pts = [i[0] for i in contour]

    # tracking corners marked
    marked = [0 for i in closed_pts]

    # compute markable corners along contour
    for ind in range(2, len(closed_pts)+2):
        i = ind%len(closed_pts)

        # POINTS LABEL
        #cv.putText(img,str(closed_pts[i]),[closed_pts[i][0],closed_pts[i][1]+(30 if i%3==0 else (-30 if i%3==1 else 0))],cv.FONT_HERSHEY_PLAIN,1,(255,255,255))
        # DRAW COLORED LINES
        annotated_lines = np.copy(img)
        cv.line(img,closed_pts[i-1],closed_pts[i],(255 if i%3==0 else 0,255 if i%3==1 else 0,255 if i%3==2 else 0),3)
        output_files.append(udisp_img(annotated_lines, "all_lines"))

        angle = getAngle(closed_pts[i-2],closed_pts[i-1],closed_pts[i])
        # ANGLE READOUT DISPLAY
        #cv.putText(img,str(angle)[:6],[closed_pts[i][0],closed_pts[i][1]+(30 if i%3==0 else (-30 if i%3==1 else 0))],cv.FONT_HERSHEY_PLAIN,1,(255,255,255))

        uprint("x  y  angle")
        uprint(str(closed_pts[i][0]) + " " +
               str(closed_pts[i][1]) + " " +
               str(angle))

        #mark points only if they are along an edge - ie a point that has ~180 degree angle
        if abs(angle-180)<10:
            uprint(f"marking {closed_pts[i]} and {closed_pts[i - 1]} and {closed_pts[i - 2]}")
            #mark the point center to the angle
            marked[i-1]=1

    # points marked are in the middle of line segs, and
    # all points not marked are parts of corners
    # thus gather points NOT MARKED
    pot_corners = []
    for i in range(len(marked)):
        if marked[i] == 0:
            pot_corners.append(closed_pts[i])

    #draw circle on the potential points
    annotated_potcorners = np.copy(img)
    for i in range(len(pot_corners)):
        #cv.putText(img, str(pot_corners[i]), [pot_corners[i][0], pot_corners[i][1]+(30 if i%3==0 else (-30 if i%3==1 else 0))], cv.FONT_HERSHEY_PLAIN, 1, (255, 255, 255))
        cv.circle(annotated_potcorners, tuple(map(int, pot_corners[i])), 10, (0, 255, 255), -1) #yellow dot

    output_files.append(udisp_img(annotated_potcorners, "potential_corners"))

    return pot_corners, output_files

def find_corners(img, potential_corners):
    """
    Given potential corners, find groups and set the middle point in each group to be the designated point for that corner
    :param img: full image
    :param potential_corners: points that potentially have corners inside
    :return: a list of corners representing final selected corners, output files
    """
    output_files = []

    annotated = np.copy(img)

    #process to find 4 corners
    # many potential corners are marked, but we wanna isolate 4 ones - the points in the center of a bunch of marked corners in a real corner
    # 250606 note: this alg is largely inefficient calculating dists and a mean across n
    #              and then looping over n + (small) k<n
    #              even tho its tech O(n) its rly like 3*n and n doesnt get super duper large so yeah o well
    corner_start,corner_end = -1,-1
    corner_indexes = set()
    dists = []
    #find distances between points to isolate those that are nearby
    #find average distance as threshold for later use
    for i in range(1, len(potential_corners)):
        dists.append(np.linalg.norm(np.array(potential_corners[i]) - np.array(potential_corners[i - 1])))
    dists.sort()
    avg = np.mean(dists[0:int(len(dists) * 3 / 4)])
    # im p sure statistically, the outliers are so big that if we avg the lower 3 quartiles we are still chill
    #plt.plot(dists, "ro")
    #plt.show()

    ind, offset = 0,0
    while ind < len(potential_corners)+offset+1:
        # i is the modded version to allow loopback of the loop and guarantee a full cycle where we hit every corner
        # ind is the actual loop, and its used when doing ind calc for accurate calcs w/o loopback issue
        # +1 required cuz the alg works by running midpt calc when u reach i+1 (cuz it then calcs curr index - 1 for midpt)
        i = ind%len(potential_corners)

        #get the vector betw pts, then get their magnitude (length) with linalg.norm
        curr_dist = np.linalg.norm(np.array(potential_corners[i]) - np.array(potential_corners[i - 1]))
        uprint(
            f'''point {potential_corners[i]}\ndist {curr_dist}''')

        # FIND when distance between adj points is large in order to isolate groups of points at a corner
        if curr_dist - avg > 10:
            # when we encounter small dist, its like 0.8 - 0.5 (curr_dist - avg) --> less than 5 fo sho
            # if the curr dist is small, then minus avg will guarantee negative
            #means that 10 is lowk arbitrary and just needs to check for negative
            # thus if its >5, then its a large distance
            if corner_start==-1:
                # if start == -1, then establish st point on encountering a large dist
                corner_start=ind
                uprint(f"starting at {corner_start}")
                offset=ind

            else:
                # now once you find another large dist, there will be a corner inside
                # (ie if large dist and start != -1)

                #   calc midpt - to i-1
                midpt = int(((ind-1)+corner_start)/2)%len(potential_corners)
                uprint(f"midpt at {midpt}")

                # add calc'd corner to list of corners (tracked by index)
                # its a set btw - ensures no duplicates
                corner_indexes.add(midpt)
                #draw circle on the finalized corner
                cv.circle(annotated, tuple(map(int, potential_corners[midpt])), 10, (255, 0, 0), -1)  # blue dot

                #   set start to i
                corner_start = ind
                uprint(f"start again at {i}")
        ind+=1
        uprint("\n")

    output_files.append(udisp_img(annotated, "corners"))

    return corner_indexes, output_files

def cut_to_object(og_img, potential_corners, corner_indexes):
    """
    Given corners, flatten and push frame to these 4 corners
    :param og_img: full image
    :param potential_corners: the pool of all potential corners
    :param corner_indexes: indices of the chosen corners
    :return: an img as Mat that is cut to the new corners, output files
    """
    output_files = []

    # find destination box
    corner_indexes = sorted(list(corner_indexes)) #REMEMBER: ITEMS IN CORNERS ARE INDEXES OF POT_CORNERS
    corners = [potential_corners[i] for i in corner_indexes]
    if len(corners)!=4:
        uprint("ERROR: LESS THAN 4 CORNERS WERE DETERMINED")
        return None
    else:
        corners = np.array(corners, dtype=np.float32)
        corners = order_corners(corners)
        uprint("corners are: " + " ".join([str(i) for i in corners]))
        u = corners[1] - corners[0]
        v = corners[2] - corners[1]
        w = corners[3] - corners[2]
        a = corners[0] - corners[3]
        #use the formula - it yields an aspect ratio length:width for the resized rectangle
        #UPDATE: FORMULA NO WORK
        #ratio = (1/2)*(cross2d(u,v)+cross2d(w,a))/(np.linalg.norm(u)+np.linalg.norm(v)+np.linalg.norm(w)+np.linalg.norm(a))**2
        #length = np.linalg.norm(u)
        #width = length/ratio
        #print(ratio)

        #compute destination size and mat
        length = max(np.linalg.norm(u),np.linalg.norm(w))
        width = max(np.linalg.norm(v),np.linalg.norm(a))
        dst = np.array([[0.0,0.0],[length,0.0],[length,width],[0.0,width]], dtype=np.float32)

        #get transformation
        M = cv.getPerspectiveTransform(corners, dst)
        img_resized = cv.warpPerspective(og_img, M, (int(length), int(width)))

        output_files.append(udisp_img(img_resized, "cut_img"))
        return img_resized, output_files


# processing for ocr optimization
def get_ypos_lines(thresh_img):
    """
    Find all horizontal lines of the calorie label for segmentation
    :param thresh_img: img processed into a threshold version
    :return: list of y positions, output files
    """
    output_files = []

    annotated = np.copy(thresh_img)
    annotated = cv.cvtColor(annotated,cv.COLOR_GRAY2BGR)

    edges = cv.Canny(annotated,50,200)
    output_files.append(udisp_img(edges,"contour-edges"))

    lines = cv.HoughLines(edges,1,np.pi/180,200)
    values = set()
    for line in lines:
        rho,theta = line[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        img_height = thresh_img.shape[0]

        line_straightness_threshold = 0.01
        if abs(y2-y1)/img_height < line_straightness_threshold:
            values.add(y2)

        cv.line(annotated,(x1,y1),(x2,y2),(0,0,255),3)
    output_files.append(udisp_img(annotated,"contour-lines"))

    #annotated = np.copy(thresh_img)
    #annotated = cv.cvtColor(annotated,cv.COLOR_GRAY2BGR)
    #cv.drawContours(annotated,contours,-1,(0,255,0),2)
    #plot_img(annotated,False)

    return list(values), output_files

def get_cut_rows(img,line_positions):
    """
    Given y positions of the lines in a calorie label table, cut up the given image to get a box for each row of the calorie label
    :param img: processed, cut image
    :param line_positions: y positions of rows
    :return: cut up rows based on given y positions, output files
    """
    output_files = []

    line_positions = sorted(line_positions)

    rows = []
    height = img.shape[0]
    for i in range(len(line_positions) - 1):

        row_size_offset_threshold = 0.005
        offset = int(height*row_size_offset_threshold)

        top = line_positions[i]
        bottom = line_positions[i+1]

        abs_row_height_threshold = 20
        if abs(top-bottom) < abs_row_height_threshold:
            # if the row is too short, dont consider it
            continue

        top -= offset
        bottom += offset
        width_constraint_threshold = 10

        row_crop = img[top:bottom, width_constraint_threshold:-width_constraint_threshold]
        rows.append(row_crop)

    return rows, output_files

def process_img_to_thresh(img):
    """
    Given OG image, process it for ocr: scaling up, gray scale, denoise, adaptive threshold
    :param img: the resized image to the label that needs to be processed
    :return: processed image, output files
    """
    output_files = []

    # size up for better ocr
    scale_factor = 3.0
    new_size = (int(img.shape[1] * scale_factor), int(img.shape[0] * scale_factor))
    resized_img = cv.resize(img, new_size, interpolation=cv.INTER_CUBIC)

    # turn image grayscale
    gray = cv.cvtColor(resized_img, cv.COLOR_BGR2GRAY)
    output_files.append(udisp_img(gray,"gray"))

    # apply blur
    #blur = cv.GaussianBlur(gray,(5,5),0)

    #upscaled image is inherently noisier - remove large noise
    denoise = cv.fastNlMeansDenoising(gray, h=17)

    # binarize image
    thresh = cv.adaptiveThreshold(denoise,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY,11,2)
    #_,otsuThresh = cv.threshold(denoise,0,255,cv.THRESH_BINARY+cv.THRESH_OTSU)
    output_files.append(udisp_img(thresh,"processed_threshold"))

    return thresh




# main functions

def get_main_objects(img):
    """
    Given og main image, find objects that could be the label
    :param img: og full image
    :return: contours that could be label, output files
    """
    output_files = []

    edges = cv.Canny(img,100,200)
    output_files.append(udisp_img(edges,"edges"))

    annotated_contours = np.copy(img)
    contours, _ = cv.findContours(edges.copy(), cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    cv.drawContours(annotated_contours,contours,-1,(0,255,0),2)
    output_files.append(udisp_img(annotated_contours,"all_contours"))

    all_objs = [] #all contours
    all_objs_areas = [] # areas of all contours
    pos_objs = [] # biggest contours
    for c in contours:
        closed = cv.convexHull(c)
        closed_area = cv.contourArea(closed)
        all_objs.append(closed)
        all_objs_areas.append(closed_area)
        #for now, use 10e4; problems in future may arise where its necessary to access the top 5 or so biggest contours
        if closed_area > 1e4:
            pos_objs.append(closed)

    if len(pos_objs)==0:
        uprint("WARN: no significantly large contours detected")

    #take the first 5 biggest objects instead
    annotated_final = np.copy(img)
    swaps=list(range(len(all_objs_areas)))
    swaps.sort(key=lambda ind: all_objs_areas[ind], reverse=True)
    for i in range(5):
        pos_objs.append(all_objs[swaps[i]])

    cv.drawContours(annotated_final, pos_objs, -1, (0,255,0), 2)
    output_files.append(udisp_img(annotated_final,"major_object_contours"))

    return pos_objs, output_files

def get_valid_contours(img, possible_objs):
    """
    Process objects to find whether they are the actual label and have data able to be scanned
    :param img: og full image
    :param possible_objs: objects that could be label
    :return: contours that work to be final label, output files
    """
    output_files = []

    valid_contours = []

    for ob in possible_objs:
        #ob is a contour
        uprint(f"{BLUE}finding potential corners{RESET}")
        pot_corners, out_mark_corners = mark_corners(img,ob)
        for outfile in out_mark_corners: output_files.append(outfile)

        uprint(f"{BLUE}getting actual corners{RESET}")
        corner_inds, out_find_corners = find_corners(img,pot_corners)
        for outfile in out_find_corners: output_files.append(outfile)
        #uprint("corner indexes are: "+" ".join([str(i) for i in corner_inds]))

        if corner_inds is None:
            uprint("toast - no corners")

        uprint(f"{BLUE}cutting object{RESET}")
        final_img, out_cut_ob = cut_to_object(img, pot_corners, corner_inds)
        for outfile in out_cut_ob: output_files.append(outfile)
        if final_img is None:
            uprint("toast - no final image able to be cut to")
        else:
            valid_contours.append(final_img)

        uprint("\n~~~~~~~~~~~~~~~~~~~~~~~~~~~\n\n\n")

    return valid_contours, output_files

def scan_text(img):
    """
    Process img and extract nutrition information out using OCR
    :param img: cropped image >> label images
    :return: list of text detected every row of the label, output files
    """
    output_files = []

    thresh, out_processed = process_img_to_thresh(img)
    for outfile in out_processed: output_files.append(outfile)

    # get horizontal lines
    line_pos, out_ypos_lines = get_ypos_lines(thresh)
    rows, out_cut_rows = get_cut_rows(thresh,line_pos)
    for outfile in out_ypos_lines: output_files.append(outfile)
    for outfile in out_cut_rows: output_files.append(outfile)

    detected_text = []
    for i in range(len(rows)):
        # fill in gaps by shaving back the whites with erosion
        kernel = np.ones((2,2),np.uint8)
        erode = cv.erode(rows[i],kernel,iterations = 1)
        output_files.append(udisp_img(erode,"eroded_back_image"))

        final = erode
        detected_text.append(pytesseract.image_to_string(final))

    uprint("finished everything")

    return detected_text, output_files


'''
im = cv.imread("upload.png")

cv.imshow("img",im)
while 1:
    if cv.waitKey(1) == ord("q"):
        break
'''