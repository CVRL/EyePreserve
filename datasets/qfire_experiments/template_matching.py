# import the necessary packages
import numpy as np
import argparse
import imutils
import glob
import cv2
from PIL import Image
import math
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

def crop_left_eye(imagePath, template, illum, high_thres_ids):
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
    # loop over the scales of the image
    for scale in np.linspace(0.9, 1.1, 3)[::-1]:
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
            thres_inc = 40
        else:
            thres_inc = 0
        if illum == 'H':
            edged = np.uint8(np.where(resized<160+thres_inc, 0, 255))
            edged = cv2.blur(edged, (5,5), 0)
        elif illum == 'M':
            edged = np.uint8(np.where(resized<130+thres_inc, 0, 255))
            edged = cv2.blur(edged, (5,5), 0)
        elif illum == 'L':
            edged = np.uint8(np.where(resized<100+thres_inc, 0, 255))
            edged = cv2.blur(edged, (3,3), 0)
        #edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCORR_NORMED)
        maxTupleL, maxTupleR = process_template_output(result, 'left')
        #print(maxTuple1, maxTuple2)
        maxValR = maxTupleR[0]
        maxLocR = maxTupleR[1]
        maxVal =  maxValR

        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLocR, r)

	# unpack the bookkeeping variable and compute the (x, y) coordinates
	# of the bounding box based on the resized ratio
    (_, maxLocR, r) = found
    w_half = 2.2 * 1.3 * (tW * r)
    h_half = (3/4)*w_half

	# draw a bounding box around the detected result and display the image
    (startX2, startY2) = (int(maxLocR[0] * r), int(maxLocR[1] * r))
    (endX2, endY2) = (int((maxLocR[0] + tW) * r), int((maxLocR[1] + tH) * r))

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

def crop_right_eye(imagePath, template, illum, high_thres_ids):
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
    # loop over the scales of the image
    for scale in np.linspace(0.1, 1.0, 20)[::-1]:
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
            thres_inc = 40
        else:
            thres_inc = 0
        if illum == 'H':
            edged = np.uint8(np.where(resized<160+thres_inc, 0, 255))
            edged = cv2.blur(edged, (5,5), 0)
        elif illum == 'M':
            edged = np.uint8(np.where(resized<130+thres_inc, 0, 255))
            edged = cv2.blur(edged, (5,5), 0)
        elif illum == 'L':
            edged = np.uint8(np.where(resized<100+thres_inc, 0, 255))
            edged = cv2.blur(edged, (3,3), 0)
        #edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCORR_NORMED)
        maxTupleL, maxTupleR = process_template_output(result, 'right')
        #print(maxTuple1, maxTuple2)
        maxValL = maxTupleL[0]
        maxLocL = maxTupleL[1]
        maxVal = maxValL

        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLocL, r)

	# unpack the bookkeeping variable and compute the (x, y) coordinates
	# of the bounding box based on the resized ratio
    (_, maxLocL, r) = found
    w_half = 2.2 * 1.3 * (tW * r)
    h_half = (3/4)*w_half
    (startX1, startY1) = (int(maxLocL[0] * r), int(maxLocL[1] * r))
    (endX1, endY1) = (int((maxLocL[0] + tW) * r), int((maxLocL[1] + tH) * r))
    

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

def crop_left_and_right_eye(imagePath, template, illum, high_thres_ids, rotate=True):
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
    # loop over the scales of the image
    for scale in np.linspace(0.1, 1.0, 20)[::-1]:
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
            thres_inc = 40
        else:
            thres_inc = 0
        if illum == 'H':
            edged = np.uint8(np.where(resized<160+thres_inc, 0, 255))
            edged = cv2.blur(edged, (5,5), 0)
        elif illum == 'M':
            edged = np.uint8(np.where(resized<130+thres_inc, 0, 255))
            edged = cv2.blur(edged, (5,5), 0)
        elif illum == 'L':
            edged = np.uint8(np.where(resized<100+thres_inc, 0, 255))
            edged = cv2.blur(edged, (3,3), 0)
        #edged = cv2.Canny(resized, 50, 200)
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
    w_half = 2.2 * 1.3 * (tW * r)
    h_half = (3/4)*w_half
    (startX1, startY1) = (int(maxLocL[0] * r), int(maxLocL[1] * r))
    (endX1, endY1) = (int((maxLocL[0] + tW) * r), int((maxLocL[1] + tH) * r))
    

	# draw a bounding box around the detected result and display the image
    (startX2, startY2) = (int(maxLocR[0] * r), int(maxLocR[1] * r))
    (endX2, endY2) = (int((maxLocR[0] + tW) * r), int((maxLocR[1] + tH) * r))

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
    pad_top = max(abs(min(startYL, 0)), abs(min(startYR, 0)))
    pad_right = max(endXL - image.shape[1], 0)
    pad_bottom = max(max(endYL - image.shape[0], 0), max(endYR - image.shape[0], 0))

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
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--template", required=True, help="Path to template image")
ap.add_argument("-i", "--images", required=True,
    help="Path to images where template will be matched")
ap.add_argument("-v", "--visualize",
    help="Flag indicating whether or not to visualize each iteration")
args = vars(ap.parse_args())
# load the image image, convert it to grayscale, and detect edges
template = cv2.imread(args["template"])
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
template = np.uint8(np.where(template<80, 0, 255))
template = cv2.blur(template, (5,5), 0)
#template = cv2.Canny(template, 50, 200)
(tH, tW) = template.shape[:2]
cv2.imshow("Template", template)
cv2.imwrite('template_processed.png', template)
best_edged = None
# loop over the images to find the template in
for imagePath in glob.glob(args["images"] + "/*.png"):
    # load the image, convert it to grayscale, and initialize the
    # bookkeeping variable to keep track of the matched region
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    #gray = np.where(gray<255, 0, gray)
    found = None
    # loop over the scales of the image
    for scale in np.linspace(0.8, 1.2, 10)[::-1]:
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
        edged = np.uint8(np.where(resized<80, 0, 255))
        edged = cv2.blur(edged, (5, 5), 0)
        #edged = cv2.Canny(resized, 50, 200)
        result = cv2.matchTemplate(edged, template, cv2.TM_CCORR_NORMED)
        maxTupleL, maxTupleR = process_template_output(result)
        #print(maxTuple1, maxTuple2)
        maxValL = maxTupleL[0]
        maxLocL = maxTupleL[1]
        maxValR = maxTupleR[0]
        maxLocR = maxTupleR[1]
        maxVal = (maxValL + maxValR)/2
        #(_, maxValCheck, _, maxLocCheck) = cv2.minMaxLoc(result)
        # check to see if the iteration should be visualized

        # if we have found a new maximum correlation value, then update
        # the bookkeeping variable
        if found is None or maxVal > found[0]:
            best_edged = edged
            #print(maxVal, maxValCheck, maxLoc1, maxLocCheck)
            found = (maxVal, maxLocL, maxLocR, r)

	# unpack the bookkeeping variable and compute the (x, y) coordinates
	# of the bounding box based on the resized ratio
    (_, maxLocL, maxLocR, r) = found
    aspect_ratio = (best_edged.shape[0]/best_edged.shape[1])
    cv2.imshow("Processed image", cv2.resize(best_edged, (800,  int(aspect_ratio * 800)), interpolation = cv2.INTER_AREA))
    cv2.imwrite('processed_image.png', best_edged)
    #print(maxLocL, maxLocR)
    w_half = 2.6 * 1.3 * (tW * r)
    h_half = (3/4)*w_half
    (startX1, startY1) = (int(maxLocL[0] * r), int(maxLocL[1] * r))
    (endX1, endY1) = (int((maxLocL[0] + tW) * r), int((maxLocL[1] + tH) * r))
    

	# draw a bounding box around the detected result and display the image
    (startX2, startY2) = (int(maxLocR[0] * r), int(maxLocR[1] * r))
    (endX2, endY2) = (int((maxLocR[0] + tW) * r), int((maxLocR[1] + tH) * r))

    image_vistemplate = np.copy(image)
    image_vistemplate = cv2.rectangle(image_vistemplate, (startX1, startY1), (endX1, endY1), (0, 0, 255), 2)
    image_vistemplate = cv2.rectangle(image_vistemplate, (startX2, startY2), (endX2, endY2), (255, 0, 0), 2)
    cv2.imwrite('matched_template.png', image_vistemplate)
    aspect_ratio = (image_vistemplate.shape[0]/image_vistemplate.shape[1])
    im_resized = cv2.resize(image_vistemplate, (800, int(aspect_ratio * 800)), interpolation = cv2.INTER_AREA)
    cv2.imshow("Image showing matched template", im_resized)

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

    (startXR, startYR) = (int(center1[0] - w_half), int(center1[1] - h_half))
    (endXR, endYR) = (int(center1[0] + w_half), int(center1[1] + h_half))
    (startXL, startYL) = (int(center2[0] - w_half), int(center2[1] - h_half))
    (endXL, endYL) = (int(center2[0] + w_half), int(center2[1] + h_half))
  
    rotated_image = Image.fromarray(image).rotate((rot_angle * 180)/math.pi, fillcolor=(127,127,127))
    image = np.array(rotated_image)

    # Pad image to take care of crops outside of image
    if startXR < 0:
        pad_left = abs(startXR)
    else:
        pad_left = 0
    if startYR < 0:
        pad_top = abs(startYR)
    else:
        pad_top = 0
    if endXL > image.shape[1]:
        pad_right = endXL - image.shape[1]
    else:
        pad_right = 0
    if endYL > image.shape[0]:
        pad_bottom = endYL - image.shape[0]
    else:
        pad_bottom = 0
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
    
    cv2.imwrite('rotated_image.png', image)
    aspect_ratio = (image.shape[0]/image.shape[1])
    im_resized = cv2.resize(image, (800, int(image.shape[0]/image.shape[1] * 800)), interpolation = cv2.INTER_AREA)
    cv2.imshow("Rotated Image", im_resized)

	# draw a bounding box around the detected result and display the image
    image = cv2.rectangle(image, (startXL, startYL), (endXL, endYL), (0, 0, 255), 2)
    image = cv2.rectangle(image, (startXR, startYR), (endXR, endYR), (0, 0, 255), 2)
    cv2.imwrite('image_with_crop_boxes.png', image)
    im_resized = cv2.resize(image, (800, int(image.shape[0]/image.shape[1] * 800)), interpolation = cv2.INTER_AREA)
    cv2.imshow("Image with crop boxes", im_resized)
    cv2.waitKey(0)