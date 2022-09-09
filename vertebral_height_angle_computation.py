# Import libraries
import numpy as np
import scipy
import matplotlib.pyplot as plt
import math
import os
import scipy.ndimage
import cv2  
import pandas as pd

# Define functions

## Angle of each plate
def horizontal_angle(p1,p2):
    xDiff = p2[0] - p1[0]
    yDiff = p2[1] - p1[1]
    return math.degrees(math.atan2(yDiff, xDiff))

# Find 2 closest points to a reference
def two_closest(points, ref):
    row_dist = points[:,1] - ref
    sorted_idx = np.argsort(np.abs(row_dist))
    return points[sorted_idx][:2]

# Distance between 2 points
def dist(p, q):
    #Return the Euclidean distance between points p and q.
    return math.hypot(p[0] - q[0], p[1] - q[1])

# *** Personalize this code to retrieve the paths of the masks corresponding
# to a vertebra and store them inside the variable ***

# Define paths
root_path = os.getcwd().replace("\\", "/") + "/In_vitro/"
out_path = os.getcwd().replace("\\", "/") + "/Processing/CT/"

# Define empty variables to store paths
patients = []
profile_L1_paths = []; profile_L2_paths = []; profile_L3_paths = []; profile_L4_paths = []

# Retrieve patient labels
for subdir, dirs, files in os.walk(root_path):
    dirs[:] = [d for d in dirs if d not in ['Incomplete']]
    for direct in dirs:
        if "PA" in direct:
            patients.append(direct)

# Retrieve paths of masks.
# If for a certain patient there is no mask for that vertebra, " " is written on the corresponding position of the array.
for patient in patients:
    directory = root_path + str(patient) + "/DRR/"
    path = directory + "masque_lat_L1.tiff"
    if os.path.exists(path):
        profile_L1_paths.append(path)
    else: profile_L1_paths.append(" ")
    path = directory + "masque_lat_L2.tiff"
    if os.path.exists(path):
        profile_L2_paths.append(path)
    else: profile_L2_paths.append(" ")
    path = directory + "masque_lat_L3.tiff"
    if os.path.exists(path):
        profile_L3_paths.append(path)
    else: profile_L3_paths.append(" ")
    path = directory + "masque_lat_L4.tiff"
    if os.path.exists(path):
        profile_L4_paths.append(path)
    else: profile_L4_paths.append(" ")

# Declare variables to store the heights and angles of each vertebra
anterior_heights = np.zeros(len(profile_L1_paths))
posterior_heights = np.zeros(len(profile_L1_paths))
top_plate_angles = np.zeros(len(profile_L1_paths))
bottom_plate_angles = np.zeros(len(profile_L1_paths))

"""
Task: Detect corners and fix perspective
Source: https://stackoverflow.com/questions/64860785/opencv-using-canny-and-shi-tomasi-to-detect-round-corners-of-a-playing-card
"""
# Loop through each mask stored in profile_LX_paths (change name to choose a certain vertebra)
for k in range(0,len(profile_L1_paths)): # *** change L1 to desired vertebra
    path = profile_L1_paths[k] # *** change L1 to desired vertebra
    # If there is a mask for that vertebra for this patient:
    if path != ' ':
        # Retrieve mask
        mask = cv2.imread(path,0)
        img = mask.astype(np.uint8)
        
        # Rotate mask: for profile ones of this dataset best results seemed to be obtained with 35º clockwise
        img = scipy.ndimage.rotate(img, -35, cval=0)
        
        # Cut rectangle that fits mask
        x,y,w,h = cv2.boundingRect(img)
        img = img[y:y+h, x:x+w]

        # This code cna divide the image into tiles (a for loop would be needed to loop through them)
        # Now it only serves so that the corners are drawn in the correct coordinates
        M = img.shape[0]#//2
        N = img.shape[1]#//2
        tiles = [img[x:x+M,y:y+N] for x in range(0,img.shape[0],M) for y in range(0,img.shape[1],N)]
        im = tiles[0]
        gray = tiles[0]
        imx = im.shape[0]
        imy = im.shape[1]
        
        # Threshold image
        ret,thresh = cv2.threshold(gray,127,255,0)
        #cv2.imshow('Thresholded original',thresh)
        #cv2.waitKey(0)
    
        ## Get contours
        contours,h = cv2.findContours(thresh,cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

        corners = np.zeros((4,2))
        ## Loop through contours (in the case there are multiple)
        for cnt in contours:
            approx = []
            constant = 0.01
            
            # Find 4 corners, if they are not found, increase constant by 0.01
            while len(approx) != 4 and constant <= 1:
                approx = cv2.approxPolyDP(cnt,constant * cv2.arcLength(cnt, True), True)
                ## calculate number of vertices
                #print(len(approx))
                constant = constant + 0.01
            
            # If 4 corners were not found:
            if constant == 1: print("Could not approximate polygon.")
                
            # If 4 corners were found:
            else:
                # The following are several plots to visualize the process:
                #tmp_img = im.copy()
                #cv2.drawContours(tmp_img, [cnt], 0, (0, 255, 255), 6)
                #cv2.imshow('Contour Borders', tmp_img)
                #cv2.waitKey(0)

                #tmp_img = im.copy()
                #cv2.drawContours(tmp_img, [cnt], 0, (255, 0, 255), -1)
                #cv2.imshow('Contour Filled', tmp_img)
                #cv2.waitKey(0)

                # Make a hull arround the contour and draw it on the original image
                #tmp_img = im.copy()
                #mask = np.zeros((im.shape[:2]), np.uint8)
                #hull = cv2.convexHull(cnt)
                #cv2.drawContours(mask, [hull], 0, (255, 255, 255), -1)
                #cv2.imshow('Convex Hull Mask', mask)
                #cv2.waitKey(0)

                # Draw minimum area rectangle
                #tmp_img = im.copy()
                #rect = cv2.minAreaRect(cnt)
                #box = cv2.boxPoints(rect)
                #box = np.int0(box)
                #cv2.drawContours(tmp_img, [box], 0, (0, 0, 255), 2)
                #cv2.imshow('Minimum Area Rectangle', tmp_img)
                #cv2.waitKey(0)

                # Draw bounding rectangle
                #tmp_img = im.copy()
                #x, y, w, h = cv2.boundingRect(cnt)
                #cv2.rectangle(tmp_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #cv2.imshow('Bounding Rectangle', tmp_img)
                #cv2.waitKey(0)

                # Bounding Rectangle and Minimum Area Rectangle
                #tmp_img = im.copy()
                #rect = cv2.minAreaRect(cnt)
                #box = cv2.boxPoints(rect)
                #box = np.int0(box)
                #cv2.drawContours(tmp_img, [box], 0, (0, 0, 255), 2)
                #x, y, w, h = cv2.boundingRect(cnt)
                #cv2.rectangle(tmp_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                #cv2.imshow('Bounding Rectangle', tmp_img)
                #cv2.waitKey(0)
                #cv2.destroyAllWindows()
                
                # Determine the most extreme points along the contour
                # https://www.pyimagesearch.com/2016/04/11/finding-extreme-points-in-contours-with-opencv/
                extLeft_tmp = tuple(cnt[cnt[:, :, 0].argmin()][0])
                extRight_tmp = tuple(cnt[cnt[:, :, 0].argmax()][0])
                extTop_tmp = tuple(cnt[cnt[:, :, 1].argmin()][0])
                extBot_tmp = tuple(cnt[cnt[:, :, 1].argmax()][0])
                
                print("Corner Points: ", extLeft_tmp, extRight_tmp, extTop_tmp, extBot_tmp)
                
                #Save corner points in variable "corners"
                corners[0][0] = extLeft_tmp[0]
                corners[0][1] = extLeft_tmp[1]
                corners[1][0] = extTop_tmp[0]
                corners[1][1] = extTop_tmp[1]
                corners[2][0] = extRight_tmp[0]
                corners[2][1] = extRight_tmp[1]
                corners[3][0] = extBot_tmp[0]
                corners[3][1] = extBot_tmp[1]
                
        # Create temporal image to show where the corner points were detected
        tmp_img = im.copy()
        for x,y in corners:
            #print(x,y)
            cv2.circle(tmp_img, (int(x),int(y)), 8, (128, 128, 128), -1)

        # Plot corners found on mask
        plt.figure(k+1)
        plt.imshow(tmp_img,cmap="gray")
        plt.show()
        
        # Define which corners correspond to the anterior part and which correspond to the posterior
        anterior = np.asarray([corners[0,:],corners[1,:]])
        posterior = np.asarray([corners[2,:],corners[3,:]])
        
        # Compute anterior and posterior distances
        # According to EOS imaging, pixel size = 0.18 mm
        anterior_heights[k] = dist(anterior[0,:], anterior[1,:])*0.18
        #print("Anterior height (mm):", anterior_heights[i])
        posterior_heights[k] = dist(posterior[0,:], posterior[1,:])*0.18
        #print("Posterior height (mm):", posterior_heights[i])
        
        # Compute angles with the horizontal. Since the image has been rotated 35 degrees, we subtract these
        top_plate_angles[k] = horizontal_angle(anterior[1,:],posterior[0,:])-35
        bottom_plate_angles[k] = horizontal_angle(anterior[0,:],posterior[1,:])-35
        
    # For masks that are not available for a certain vertebra of a patient, the value recorded will be 0:
    else: continue;

# Save heights and angles to Excel file
df = pd.DataFrame({'Patient': patients, 'Anterior Height': anterior_heights, 'Posterior Height': posterior_heights, 'Top Plate Angle': top_plate_angles, 'Bottom Plate Angle': bottom_plate_angles})
df.to_excel(out_path + "profile_heights_angles_L1.xlsx") # *** change L1 to desired vertebra