import shutil
import os
from tqdm import tqdm
import pandas as pd

with open('Exclude_QFIRE_Error.txt', 'r') as qfireexclusions:
    QFIRE_exclusions = []
    for line in qfireexclusions:
        QFIRE_exclusions.append(line.strip())

with open('Exclude_MBARK_Error.txt', 'r') as mbarkexclusions:
    MBARK_exclusions = []
    for line in mbarkexclusions:
        MBARK_exclusions.append(line.strip())

dest_loc = './Q-FIRE_dataset/'
if not os.path.exists(dest_loc):
    os.mkdir(dest_loc)
source_mbark = './Q-FIRE_dataset_MBARK/'
source_qfire = './Q-FIRE_dataset_QFIRE/'

mbark_metadata = pd.read_csv("./Q-FIRE_dataset_MBARK/Q-FIRE-info-mbark.csv")
mbark_metadata.columns = [c.replace(' ', '_').lower() for c in mbark_metadata.columns]
qfire_metadata = pd.read_csv("./Q-FIRE_dataset_QFIRE/Q-FIRE-info-nonmbark.csv")
qfire_metadata.columns = [c.replace(' ', '_').lower() for c in qfire_metadata.columns]

for id in tqdm(os.listdir(source_mbark), total=len(list(os.listdir(source_mbark)))):
    id_folder = os.path.join(source_mbark, id)
    if not os.path.isdir(id_folder):
        continue
    dest_id_folder = os.path.join(dest_loc, id)
    if not os.path.exists(dest_id_folder):
        os.mkdir(dest_id_folder)
    for imagename in os.listdir(id_folder):
        if not imagename in MBARK_exclusions:
            shutil.copy(os.path.join(id_folder, imagename), os.path.join(dest_id_folder, imagename))
        else:
            print('\nSkipping', imagename, 'from MBARK')
            key = imagename.split('_')[0] + '/' + imagename
            mbark_metadata = mbark_metadata[mbark_metadata.image_location != key]

for id in tqdm(os.listdir(source_qfire), total=len(list(os.listdir(source_qfire)))):
    id_folder = os.path.join(source_qfire, id)
    if not os.path.isdir(id_folder):
        continue
    dest_id_folder = os.path.join(dest_loc, id)
    if not os.path.exists(dest_id_folder):
        os.mkdir(dest_id_folder)
    for imagename in os.listdir(id_folder):
        if not imagename in QFIRE_exclusions:
            shutil.copy(os.path.join(id_folder, imagename), os.path.join(dest_id_folder, imagename))
        else:
            print('\nSkipping', imagename, 'from Non-MBARK')
            key = imagename.split('_')[0] + '/' + imagename
            qfire_metadata = qfire_metadata[qfire_metadata.image_location != key]

final_metadata = pd.concat([qfire_metadata, mbark_metadata])
final_metadata.to_csv(os.path.join(dest_loc, 'info-full.csv'), index=False)