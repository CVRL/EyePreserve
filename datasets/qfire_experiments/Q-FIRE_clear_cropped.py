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
from imutils.object_detection import non_max_suppression

'''
left_eye, right_eye = crop_left_and_right_eye('./Q-FIRE_clear/Q-FIRE_Visit1/0127344_1/0127344_illumination_05_ft_H.png', template)
cv2.imwrite('left_eye.png', left_eye)
cv2.imwrite('right_eye.png', right_eye)

'''
folder_pre = './Q-FIRE_clear/Q-FIRE_Visit'
dest_folder = './Q-FIRE_clear_cropped/Q-FIRE_Visit'
if not os.path.exists('./Q-FIRE_clear_cropped/'):
    os.mkdir('./Q-FIRE_clear_cropped/')

#print(dataset_df.columns)

for visit in ['1', '2']:
    print('Processing Visit', visit)
    folder = folder_pre + visit
    for id_visit in tqdm(os.listdir(folder), total=len(list(os.listdir(folder)))):
        if not os.path.exists(dest_folder+visit):
            os.mkdir(dest_folder+visit)
        dest_id_folder = os.path.join(dest_folder+visit, id_visit)
        if not os.path.exists(dest_id_folder):
            os.mkdir(dest_id_folder)
        id_folder = os.path.join(folder, id_visit)
        for imagename in os.listdir(id_folder):
            imagename_parts = imagename.split('.')[0].split('_')
            dist = int(imagename_parts[2])
            imagePath = os.path.join(id_folder, imagename)
            image_uncropped = cv2.imread(imagePath)
            non_empty_columns = np.where(image_uncropped.mean(axis=0)>40)[0]
            non_empty_rows = np.where(image_uncropped.mean(axis=1)>40)[0]
            cropBox = (min(non_empty_rows), max(non_empty_rows), min(non_empty_columns), max(non_empty_columns))
            image = image_uncropped[cropBox[0]:cropBox[1]+1, cropBox[2]:cropBox[3]+1]
            if dist == 11:
                image = image[0:int(3*image.shape[0]/4), :]
            cv2.imwrite(os.path.join(dest_id_folder, imagename), image)
        #print(dataset_df.head())