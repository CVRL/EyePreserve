# import the necessary packages
import numpy as np
import argparse
import imutils
import glob
import cv2
from PIL import Image
import math
import pandas as pd
import os
from tqdm import tqdm
'''
def process_template_output(result):
    tuple_list_L = []
    half_col = int(result.shape[1]/2)
    for i in range(result.shape[0]):
        for j in range(half_col):
            tuple = (result[i,j], (j, i))
            tuple_list_L.append(tuple)
    max_tuple_L = max(tuple_list_L, key=lambda x: x[0])
    tuple_list_R = []
    for i in range(result.shape[0]):
        for j in range(half_col, result.shape[1]):
            tuple = (result[i,j], (j, i))
            tuple_list_R.append(tuple)
    max_tuple_R = max(tuple_list_R, key=lambda x: x[0])
    return max_tuple_L, max_tuple_R
'''

def process_template_output(result, eye='both'):
    if eye == 'both':
        check_row = int(result.shape[0]/5)
        half_col = int(result.shape[1]/2)
        (_, val_L1, _, loc_L1) = cv2.minMaxLoc(result[:, :half_col])
        (_, val_R1, _, loc_R1) = cv2.minMaxLoc(result[loc_L1[1]-check_row:loc_L1[1]+check_row, half_col:])
        val1 = val_L1 + val_R1

        (_, val_R2, _, loc_R2) = cv2.minMaxLoc(result[:, half_col:])
        (_, val_L2, _, loc_L2) = cv2.minMaxLoc(result[loc_R2[1]-check_row:loc_R2[1]+check_row, :half_col])
        val2 = val_L2 + val_R2
        
        if val1 >= val2:
            maxVal_L = val_L1
            maxLoc_L = loc_L1
            maxVal_R = val_R1
            loc_R1 = (loc_R1[0], loc_R1[1]+loc_L1[1]-check_row)
            maxLoc_R = loc_R1
        else:
            maxVal_L = val_L2
            loc_L2 = (loc_L2[0], loc_L2[1]+loc_R2[1]-check_row)
            maxLoc_L = loc_L2
            maxVal_R = val_R2
            maxLoc_R = loc_R2
        
        #(_, maxVal_R, _, maxLoc_R)  = cv2.minMaxLoc(result[:, half_col:])
        maxLoc_R = (maxLoc_R[0]+(half_col), maxLoc_R[1])
        return (maxVal_L, maxLoc_L), (maxVal_R, maxLoc_R)
    elif eye == 'left':
        half_col = int(result.shape[1]/2)
        (_, maxVal_R, _, maxLoc_R) = cv2.minMaxLoc(result[:, half_col:])
        maxLoc_R = (maxLoc_R[0]+(half_col), maxLoc_R[1])
        return (None, None), (maxVal_R, maxLoc_R)
    elif eye == 'right':
        half_col = int(result.shape[1]/2)
        (_, maxVal_L, _, maxLoc_L) = cv2.minMaxLoc(result[:, :half_col])
        return (maxVal_L, maxLoc_L), (None, None)

def add_margin(pil_img, pad):
    width, height = pil_img.size
    new_width = width + 2 * pad
    new_height = height + 2 * pad
    result = Image.new(pil_img.mode, (new_width, new_height), (127, 127,127))
    result.paste(pil_img, (pad, pad))
    return result

def crop_left_eye(imagePath, template, dist, illum, upper_crop_ids, low_thres_ids, high_thres_ids, debug=False):
    # load the template image and process it
    (tH, tW) = template.shape[:2]
    # load the image, convert it to grayscale, and initialize the
    # bookkeeping variable to keep track of the matched region
    #id = imagePath.split('/')[-2].split('_')[0]
    id = os.path.split(imagePath)[-1].split('_')[0]
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray = np.where(gray<255, 0, gray)
    found = None
    best_edged = None
    # loop over the scales of the image
    for scale in np.linspace(0.8, 1.2, 5)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])
        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        if id in high_thres_ids:
            thres_inc = 60
            thres_dec = 0
        elif id in low_thres_ids:
            thres_inc = 0
            thres_dec = -30
        else:
            thres_inc = 0
            thres_dec = 0
        if illum == 'H':
            edged = np.uint8(np.where(resized<140+thres_inc+thres_dec, 0, 255))
        elif illum == 'M':
            edged = np.uint8(np.where(resized<90+thres_inc+thres_dec, 0, 255))
        elif illum == 'L':
            edged = np.uint8(np.where(resized<80+thres_inc+thres_dec, 0, 255))
        if dist in ['11ft', '25ft']:
            edged = cv2.blur(edged, (3,3), 0)
        else:
            edged = cv2.blur(edged, (5,5), 0)
        #edged = cv2.Canny(resized, 50, 200)
        if id in upper_crop_ids:
            result = cv2.matchTemplate(edged[0:int(3*edged.shape[0]/4), :], template, cv2.TM_CCORR_NORMED)
        else:
            result = cv2.matchTemplate(edged, template, cv2.TM_CCORR_NORMED)
        maxTupleL, maxTupleR = process_template_output(result, 'left')
        #print(maxTuple1, maxTuple2)
        maxValR = maxTupleR[0]
        maxLocR = maxTupleR[1]
        maxVal =  maxValR

        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            best_edged = edged
            found = (maxVal, maxLocR, r)

	# unpack the bookkeeping variable and compute the (x, y) coordinates
	# of the bounding box based on the resized ratio
    (_, maxLocR, r) = found
    w_half = 2.6 * 1.3 * (tW * r)
    h_half = (3/4)*w_half

	# draw a bounding box around the detected result and display the image
    (startX2, startY2) = (int(maxLocR[0] * r), int(maxLocR[1] * r))
    (endX2, endY2) = (int((maxLocR[0] + tW) * r), int((maxLocR[1] + tH) * r))

    if debug:
        aspect_ratio = (best_edged.shape[0]/best_edged.shape[1])
        cv2.imshow("Processed image", cv2.resize(best_edged, (800,  int(aspect_ratio * 800)), interpolation = cv2.INTER_AREA))
        image_vistemplate = np.copy(image)
        image_vistemplate = cv2.rectangle(image_vistemplate, (startX2, startY2), (endX2, endY2), (255, 0, 0), 2)
        aspect_ratio = (image_vistemplate.shape[0]/image_vistemplate.shape[1])
        im_resized = cv2.resize(image_vistemplate, (800, int(aspect_ratio * 800)), interpolation = cv2.INTER_AREA)
        cv2.imshow("Image showing matched template", im_resized)

    #image_vistemplate = np.copy(image)
    #image_vistemplate = cv2.rectangle(image_vistemplate, (startX1, startY1), (endX1, endY1), (0, 0, 255), 2)
    #image_vistemplate = cv2.rectangle(image_vistemplate, (startX2, startY2), (endX2, endY2), (255, 0, 0), 2)
    #cv2.imwrite('matched_template.png', image_vistemplate)
    #im_resized = cv2.resize(image_vistemplate, (800, 600), interpolation = cv2.INTER_AREA)
    #cv2.imshow("Image showing matched template", im_resized)

    center2 = ((startX2 + endX2)/2, (startY2 + endY2)/2)
    
    (startXL, startYL) = (int(center2[0] - w_half), int(center2[1] - h_half))
    (endXL, endYL) = (int(center2[0] + w_half), int(center2[1] + h_half))

    # Pad image to take care of crops outside of image
    pad_right = max(endXL - image.shape[1], 0)
    pad_top = abs(min(startYL, 0))
    pad_bottom = max(endYL - image.shape[0], 0)

    max_pad = max([pad_right, pad_top, pad_bottom])
    padded_image = add_margin(Image.fromarray(image), max_pad)
    image = np.array(padded_image)
    startXL += max_pad
    endXL += max_pad
    startYL += max_pad
    endYL += max_pad

    left_eye = np.copy(image)[startYL:endYL, startXL:endXL]
    hl, wl = left_eye.shape[:2]

    #if hl == 0:
        #print('Why the fuck is this 0?')
        #print(center2, startYL, endYL, startXL, endXL, pad_right, pad_bottom)

    assert wl/hl >= 1.31 and wl/hl <= 1.35

    left_eye = cv2.resize(left_eye, (640, 480), interpolation=cv2.INTER_LINEAR)
    
    return left_eye

def crop_right_eye(imagePath, template, dist, illum, upper_crop_ids, low_thres_ids, high_thres_ids, debug=False):
    # load the template image and process it
    (tH, tW) = template.shape[:2]
    # load the image, convert it to grayscale, and initialize the
    # bookkeeping variable to keep track of the matched region
    #id = imagePath.split('/')[-2].split('_')[0]
    id = os.path.split(imagePath)[-1].split('_')[0]
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray = np.where(gray<255, 0, gray)
    found = None
    best_edged = None
    # loop over the scales of the image
    for scale in np.linspace(0.8, 1.2, 5)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])
        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        if id in high_thres_ids:
            thres_inc = 60
            thres_dec = 0
        elif id in low_thres_ids:
            thres_inc = 0
            thres_dec = -30
        else:
            thres_inc = 0
            thres_dec = 0
        if illum == 'H':
            edged = np.uint8(np.where(resized<140+thres_inc+thres_dec, 0, 255))
        elif illum == 'M':
            edged = np.uint8(np.where(resized<90+thres_inc+thres_dec, 0, 255))
        elif illum == 'L':
            edged = np.uint8(np.where(resized<80+thres_inc+thres_dec, 0, 255))
        if dist in ['11ft', '25ft']:
            edged = cv2.blur(edged, (3,3), 0)
        else:
            edged = cv2.blur(edged, (5,5), 0)
        #edged = cv2.Canny(resized, 50, 200)
        #edged = cv2.Canny(resized, 50, 200)
        if id in upper_crop_ids:
            result = cv2.matchTemplate(edged[0:int(3*edged.shape[0]/4), :], template, cv2.TM_CCORR_NORMED)
        else:
            result = cv2.matchTemplate(edged, template, cv2.TM_CCORR_NORMED)
        maxTupleL, maxTupleR = process_template_output(result, 'right')
        #print(maxTuple1, maxTuple2)
        maxValL = maxTupleL[0]
        maxLocL = maxTupleL[1]
        maxVal = maxValL

        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            best_edged = edged
            found = (maxVal, maxLocL, r)

	# unpack the bookkeeping variable and compute the (x, y) coordinates
	# of the bounding box based on the resized ratio
    (_, maxLocL, r) = found
    w_half = 2.6 * 1.3 * (tW * r)
    h_half = (3/4)*w_half
    (startX1, startY1) = (int(maxLocL[0] * r), int(maxLocL[1] * r))
    (endX1, endY1) = (int((maxLocL[0] + tW) * r), int((maxLocL[1] + tH) * r))

    if debug:
        aspect_ratio = (best_edged.shape[0]/best_edged.shape[1])
        cv2.imshow("Processed image", cv2.resize(best_edged, (800,  int(aspect_ratio * 800)), interpolation = cv2.INTER_AREA))
        image_vistemplate = np.copy(image)
        image_vistemplate = cv2.rectangle(image_vistemplate, (startX1, startY1), (endX1, endY1), (0, 0, 255), 2)
        aspect_ratio = (image_vistemplate.shape[0]/image_vistemplate.shape[1])
        im_resized = cv2.resize(image_vistemplate, (800, int(aspect_ratio * 800)), interpolation = cv2.INTER_AREA)
        cv2.imshow("Image showing matched template", im_resized)
    

	# draw a bounding box around the detected result and display the image
    #image_vistemplate = np.copy(image)
    #image_vistemplate = cv2.rectangle(image_vistemplate, (startX1, startY1), (endX1, endY1), (0, 0, 255), 2)
    #image_vistemplate = cv2.rectangle(image_vistemplate, (startX2, startY2), (endX2, endY2), (255, 0, 0), 2)
    #cv2.imwrite('matched_template.png', image_vistemplate)
    #im_resized = cv2.resize(image_vistemplate, (800, 600), interpolation = cv2.INTER_AREA)
    #cv2.imshow("Image showing matched template", im_resized)

    center1 = ((startX1 + endX1)/2, (startY1 + endY1)/2)
    
    (startXR, startYR) = (int(center1[0] - w_half), int(center1[1] - h_half))
    (endXR, endYR) = (int(center1[0] + w_half), int(center1[1] + h_half))

    # Pad image to take care of crops outside of image
    pad_left = abs(min(startXR, 0))
    pad_top = abs(min(startYR, 0))
    pad_bottom = max(endYR - image.shape[0], 0)

    max_pad = max([pad_left, pad_top, pad_bottom])
    padded_image = add_margin(Image.fromarray(image), max_pad)
    image = np.array(padded_image)
    startXR += max_pad
    endXR += max_pad
    startYR += max_pad
    endYR += max_pad

    right_eye = np.copy(image)[startYR:endYR, startXR:endXR]
    hr, wr = right_eye.shape[:2]

    assert wr/hr >= 1.31 and wr/hr <= 1.35
    
    right_eye = cv2.resize(right_eye, (640, 480), interpolation=cv2.INTER_LINEAR)

    return right_eye

def crop_left_and_right_eye(imagePath, template, dist, illum, upper_crop_ids, low_thres_ids, high_thres_ids, rotate=True, debug=False):
    # load the template image and process it
    (tH, tW) = template.shape[:2]
    # load the image, convert it to grayscale, and initialize the
    # bookkeeping variable to keep track of the matched region
    #id = imagePath.split('/')[-2].split('_')[0]
    id = os.path.split(imagePath)[-1].split('_')[0]
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #gray = np.where(gray<255, 0, gray)
    found = None
    best_edged = None
    # loop over the scales of the image
    for scale in np.linspace(0.8, 1.2, 5)[::-1]:
        # resize the image according to the scale, and keep track
        # of the ratio of the resizing
        resized = imutils.resize(gray, width = int(gray.shape[1] * scale))
        r = gray.shape[1] / float(resized.shape[1])
        # if the resized image is smaller than the template, then break
        # from the loop
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        # detect edges in the resized, grayscale image and apply template
        # matching to find the template in the image
        if id in high_thres_ids:
            thres_inc = 60
            thres_dec = 0
            thres_dec2 = 0
        elif id in low_thres_ids:
            thres_inc = 0
            thres_dec = -50
            thres_dec2 = -10
        else:
            thres_inc = 0
            thres_dec = 0
            thres_dec2 = 0
        if illum == 'H':
            edged = np.uint8(np.where(resized<140+thres_inc+thres_dec, 0, 255))
        elif illum == 'M':
            edged = np.uint8(np.where(resized<90+thres_inc+thres_dec2, 0, 255))
        elif illum == 'L':
            edged = np.uint8(np.where(resized<80+thres_inc+thres_dec2, 0, 255))
        if dist in ['11ft', '25ft']:
            edged = cv2.blur(edged, (3,3), 0)
            h, w = edged.shape[:2]
        else:
            if id in low_thres_ids:
                edged = cv2.blur(edged, (7,7), 0)
            else:
                edged = cv2.blur(edged, (5,5), 0)
            
                
        #edged = cv2.Canny(resized, 50, 200)
        #edged = cv2.Canny(resized, 50, 200)
        if id in upper_crop_ids:
            result = cv2.matchTemplate(edged[0:int(3*edged.shape[0]/4), :], template, cv2.TM_CCORR_NORMED)
        else:
            result = cv2.matchTemplate(edged, template, cv2.TM_CCORR_NORMED)
        maxTupleL, maxTupleR = process_template_output(result, 'both')
        #print(maxTuple1, maxTuple2)
        maxValL = maxTupleL[0]
        maxLocL = maxTupleL[1]
        maxValR = maxTupleR[0]
        maxLocR = maxTupleR[1]
        maxVal = (maxValL + maxValR)/2

        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            best_edged = edged
            found = (maxVal, maxLocL, maxLocR, r)

	# unpack the bookkeeping variable and compute the (x, y) coordinates
	# of the bounding box based on the resized ratio
    
    (_, maxLocL, maxLocR, r) = found
    w_half = 2.6 * 1.3 * (tW * r)
    h_half = (3/4)*w_half
    (startX1, startY1) = (int(maxLocL[0] * r), int(maxLocL[1] * r))
    (endX1, endY1) = (int((maxLocL[0] + tW) * r), int((maxLocL[1] + tH) * r))
    

	# draw a bounding box around the detected result and display the image
    (startX2, startY2) = (int(maxLocR[0] * r), int(maxLocR[1] * r))
    (endX2, endY2) = (int((maxLocR[0] + tW) * r), int((maxLocR[1] + tH) * r))

    if debug:
        aspect_ratio = (best_edged.shape[0]/best_edged.shape[1])
        #print('Show the fucking image')
        cv2.imshow("Processed image", cv2.resize(best_edged, (800,  int(aspect_ratio * 800)), interpolation = cv2.INTER_AREA))
        image_vistemplate = np.copy(image)
        image_vistemplate = cv2.rectangle(image_vistemplate, (startX1, startY1), (endX1, endY1), (0, 0, 255), 2)
        image_vistemplate = cv2.rectangle(image_vistemplate, (startX2, startY2), (endX2, endY2), (255, 0, 0), 2)
        aspect_ratio = (image_vistemplate.shape[0]/image_vistemplate.shape[1])
        im_resized = cv2.resize(image_vistemplate, (800, int(aspect_ratio * 800)), interpolation = cv2.INTER_AREA)
        cv2.imshow("Image showing matched template", im_resized)
        cv2.waitKey(0)

    #image_vistemplate = np.copy(image)
    #image_vistemplate = cv2.rectangle(image_vistemplate, (startX1, startY1), (endX1, endY1), (0, 0, 255), 2)
    #image_vistemplate = cv2.rectangle(image_vistemplate, (startX2, startY2), (endX2, endY2), (255, 0, 0), 2)
    #cv2.imwrite('matched_template.png', image_vistemplate)
    #im_resized = cv2.resize(image_vistemplate, (800, 600), interpolation = cv2.INTER_AREA)
    #cv2.imshow("Image showing matched template", im_resized)

    center1 = ((startX1 + endX1)/2, (startY1 + endY1)/2)
    center2 = ((startX2 + endX2)/2, (startY2 + endY2)/2)
    

    y = center2[1] - center1[1]
    x = center2[0] - center1[0]
    rot_angle = math.atan(y/x)
    #if rot_angle < 0:
    #    rot_angle += 2 * math.pi
    
    R = np.matrix(
        [
            [math.cos(-rot_angle), -math.sin(-rot_angle)],
            [math.sin(-rot_angle), math.cos(-rot_angle)]
        ]
    )

    image_center_x = int(image.shape[1]/2)
    image_center_y = int(image.shape[0]/2)

    rotcenter1 = R @ np.matrix([center1[0] - image_center_x, center1[1] - image_center_y]).T
    rotcenter2 = R @ np.matrix([center2[0] - image_center_x, center2[1] - image_center_y]).T

    center1 = [int(rotcenter1[0, 0] + image_center_x), int(rotcenter1[1, 0] + image_center_y)]
    center2 = [int(rotcenter2[0, 0] + image_center_x), int(rotcenter2[1, 0] + image_center_y)]

    rotated_image = Image.fromarray(image).rotate((rot_angle * 180)/math.pi, fillcolor=(127,127,127))
    image = np.array(rotated_image)
    
    (startXR, startYR) = (int(center1[0] - w_half), int(center1[1] - h_half))
    (endXR, endYR) = (int(center1[0] + w_half), int(center1[1] + h_half))
    (startXL, startYL) = (int(center2[0] - w_half), int(center2[1] - h_half))
    (endXL, endYL) = (int(center2[0] + w_half), int(center2[1] + h_half))

    # Pad image to take care of crops outside of image
    pad_left = abs(min(startXR, 0))
    pad_top = abs(min(min(startYL, startYR), 0))
    pad_right = max(endXL - image.shape[1], 0)
    pad_bottom = max(max(endYL - image.shape[0], endYR - image.shape[0]), 0)

    max_pad = max([pad_left, pad_top, pad_right, pad_bottom])
    padded_image = add_margin(Image.fromarray(image), max_pad)
    image = np.array(padded_image)
    startXL += max_pad
    endXL += max_pad
    startYL += max_pad
    endYL += max_pad
    startXR += max_pad
    endXR += max_pad
    startYR += max_pad
    endYR += max_pad

    left_eye = np.copy(image)[startYL:endYL, startXL:endXL]
    right_eye = np.copy(image)[startYR:endYR, startXR:endXR]
    hl, wl = left_eye.shape[:2]
    hr, wr = right_eye.shape[:2]

    assert wl/hl >= 1.31 and wl/hl <= 1.35 and wr/hr >= 1.31 and wr/hr <= 1.35

    left_eye = cv2.resize(left_eye, (640, 480), interpolation=cv2.INTER_LINEAR)
    right_eye = cv2.resize(right_eye, (640, 480), interpolation=cv2.INTER_LINEAR)
    
    return left_eye, right_eye



# construct the argument parser and parse the arguments
'''
templatePath_H = './template_high.png'
template_H = cv2.imread(templatePath_H)
template_H = cv2.cvtColor(template_H, cv2.COLOR_BGR2GRAY)
template_H = np.uint8(np.where(template_H<160, 0, 255))
template_H = cv2.blur(template_H, (5,5), 0)
templatePath_M = './template_medium.png'
template_M = cv2.imread(templatePath_M)
template_M = cv2.cvtColor(template_M, cv2.COLOR_BGR2GRAY)
template_M = np.uint8(np.where(template_M<130, 0, 255))
template_M = cv2.blur(template_M, (5,5), 0)
templatePath_L = './template_low.png'
template_L = cv2.imread(templatePath_L)
template_L = cv2.cvtColor(template_L, cv2.COLOR_BGR2GRAY)
template_L = np.uint8(np.where(template_L<100, 0, 255))
template_L = cv2.blur(template_L, (3,3), 0)
'''

templates = {}
for illum in ['L', 'M', 'H']:
    templates[illum] = {}
    for dist in ['05ft', '07ft', '11ft', '15ft', '25ft']:
        templates[illum][dist] = cv2.imread('./template_'+illum+'_'+dist+'.png')
        templates[illum][dist] = cv2.cvtColor(templates[illum][dist], cv2.COLOR_BGR2GRAY)
        if illum == 'H':
            templates[illum][dist] = np.uint8(np.where(templates[illum][dist]<140, 0, 255))
        elif illum == 'M':
            templates[illum][dist] = np.uint8(np.where(templates[illum][dist]<90, 0, 255))
        elif illum == 'L':
            templates[illum][dist] = np.uint8(np.where(templates[illum][dist]<80, 0, 255))
        if dist in ['11ft', '25ft']:
            templates[illum][dist] = cv2.blur(templates[illum][dist], (3,3), 0)
        else:
            templates[illum][dist] = cv2.blur(templates[illum][dist], (5,5), 0)


'''
left_eye, right_eye = crop_left_and_right_eye('./Q-FIRE_clear/Q-FIRE_Visit1/0127344_1/0127344_illumination_05_ft_H.png', template)
cv2.imwrite('left_eye.png', left_eye)
cv2.imwrite('right_eye.png', right_eye)

'''
folder_pre = './Q-FIRE_clear_cropped/Q-FIRE_Visit'
dest_folder = './Q-FIRE_dataset_QFIRE/'
if not os.path.exists(dest_folder):
    os.mkdir(dest_folder)

demographics_df = pd.read_excel('Demographics.xlsx')
print(demographics_df.dtypes)

dataset_df = pd.DataFrame(columns = demographics_df.columns)
dataset_df['Image Location']=""

#print(dataset_df.columns)
only_left_list = {}
with open('Visit1_Left.txt', 'r') as visit1leftfile:
    only_left_list['1'] = []
    for line in visit1leftfile:
        only_left_list['1'].append(line.split('.')[0])

with open('Visit2_Left.txt', 'r') as visit2leftfile:
    only_left_list['2'] = []
    for line in visit2leftfile:
        only_left_list['2'].append(line.split('.')[0])

only_right_list = {}
with open('Visit1_Right.txt', 'r') as visit1rightfile:
    only_right_list['1'] = []
    for line in visit1rightfile:
        only_right_list['1'].append(line.split('.')[0])

with open('Visit2_Right.txt', 'r') as visit2rightfile:
    only_right_list['2'] = []
    for line in visit2rightfile:
        only_right_list['2'].append(line.split('.')[0])

with open('Exclude.txt', 'r') as excludefile:
    exclude_list = []
    for line in excludefile:
        exclude_list.append(line.strip())

with open('High_Threshold_ID.txt', 'r') as htFile:
    high_thres_ids = []
    for line in htFile:
        high_thres_ids.append(line)

with open('Low_Threshold_ID.txt', 'r') as ltFile:
    low_thres_ids = []
    for line in ltFile:
        low_thres_ids.append(line)

with open('Upper_Crop_ID.txt', 'r') as ltFile:
    upper_crop_ids = []
    for line in ltFile:
        upper_crop_ids.append(line)

for visit in ['1', '2']:
    print('Processing Visit', visit)
    folder = folder_pre + visit
    for id_visit in tqdm(os.listdir(folder), total=len(list(os.listdir(folder)))):
        id = id_visit.split('_')[0]
        dest_id_folder = os.path.join(dest_folder, id)
        if not os.path.exists(dest_id_folder):
            os.mkdir(dest_id_folder)
        id_folder = os.path.join(folder, id_visit)
        for imagename in os.listdir(id_folder):
            if id_visit+'/'+imagename in exclude_list:
                print('Skipping ', id_visit+'/'+imagename)
                continue
            imagePath = os.path.join(id_folder, imagename)
            imagename_parts = imagename.split('.')[0].split('_')
            dist = imagename_parts[2]+'ft'
            illum = imagename_parts[4]
            destimagename_pre1 = id + '_'
            destimagename_pre2 = '_' + dist + '_' + illum + '_visit' + visit + '.png'

            debug = False
            #if '2658752_LeftIris_05ft_H_visit1.png' == destimagename_pre1+'LeftIris'+destimagename_pre2:
            #    debug = True
            if imagename.split('.')[0] in only_left_list[visit]:
                if illum == 'H':
                    left_image = crop_left_eye(imagePath, templates[illum][dist], dist, illum, upper_crop_ids, low_thres_ids, high_thres_ids, debug=debug)
                elif illum == 'M':
                    left_image = crop_left_eye(imagePath, templates[illum][dist], dist, illum, upper_crop_ids, low_thres_ids, high_thres_ids, debug=debug)
                else:
                    left_image = crop_left_eye(imagePath, templates[illum][dist], dist, illum, upper_crop_ids, low_thres_ids, high_thres_ids, debug=debug)
                cv2.imwrite(os.path.join(dest_id_folder, destimagename_pre1+'LeftIris'+destimagename_pre2), left_image)
                demographics = demographics_df.loc[demographics_df['Subject ID'] == int(id)]
                dataset_row1 = demographics.copy()
                dataset_row1['Image Location'] = id + '/' + destimagename_pre1+'LeftIris'+destimagename_pre2
                dataset_row1['Eye'] = 'Left'
                dataset_row1['Source'] = destimagename_pre2.split('.')[0][1:]
                dataset_df = pd.concat([dataset_df, dataset_row1])
            elif imagename.split('.')[0] in only_right_list[visit]:
                if illum == 'H':
                    right_image = crop_right_eye(imagePath, templates[illum][dist], dist, illum, upper_crop_ids, low_thres_ids, high_thres_ids, debug=debug)
                elif illum == 'M':
                    right_image = crop_right_eye(imagePath, templates[illum][dist], dist, illum, upper_crop_ids, low_thres_ids, high_thres_ids, debug=debug)
                else:
                    right_image = crop_right_eye(imagePath, templates[illum][dist], dist, illum, upper_crop_ids, low_thres_ids, high_thres_ids, debug=debug)
                cv2.imwrite(os.path.join(dest_id_folder, destimagename_pre1+'RightIris'+destimagename_pre2), right_image)
                demographics = demographics_df.loc[demographics_df['Subject ID'] == int(id)]
                dataset_row2 = demographics.copy()
                dataset_row2['Image Location'] = id + '/' + destimagename_pre1+'RightIris'+destimagename_pre2
                dataset_row2['Eye'] = 'Right'
                dataset_row2['Source'] = destimagename_pre2.split('.')[0][1:]
                dataset_df = pd.concat([dataset_df, dataset_row2])
            else:
                if illum == 'H':
                    left_image, right_image = crop_left_and_right_eye(imagePath, templates[illum][dist], dist, illum, upper_crop_ids, low_thres_ids, high_thres_ids, debug=debug)
                elif illum == 'M':
                    left_image, right_image = crop_left_and_right_eye(imagePath, templates[illum][dist], dist, illum, upper_crop_ids, low_thres_ids, high_thres_ids, debug=debug)
                else:
                    left_image, right_image = crop_left_and_right_eye(imagePath, templates[illum][dist], dist, illum, upper_crop_ids, low_thres_ids, high_thres_ids, debug=debug)
                cv2.imwrite(os.path.join(dest_id_folder, destimagename_pre1+'LeftIris'+destimagename_pre2), left_image)
                cv2.imwrite(os.path.join(dest_id_folder, destimagename_pre1+'RightIris'+destimagename_pre2), right_image)
                demographics = demographics_df.loc[demographics_df['Subject ID'] == int(id)]
                dataset_row1 = demographics.copy()
                dataset_row1['Image Location'] = id + '/' + destimagename_pre1+'LeftIris'+destimagename_pre2
                dataset_row1['Eye'] = 'Left'
                dataset_row1['Source'] = destimagename_pre2.split('.')[0][1:]
                dataset_row2 = demographics.copy()
                dataset_row2['Image Location'] = id + '/' + destimagename_pre1+'RightIris'+destimagename_pre2
                dataset_row2['Eye'] = 'Right'
                dataset_row2['Source'] = destimagename_pre2.split('.')[0][1:]
                dataset_df = pd.concat([dataset_df, dataset_row1, dataset_row2])
        #print(dataset_df.head())


dataset_df.to_csv(os.path.join(dest_folder, 'Q-FIRE-info-nonmbark.csv'), index=False)