# This is python code for splitting the image in a folder into their categories folder

import os
import re
import shutil
DIRECTORY = r"C:\Users\RaihanSatar\OneDrive\Desktop\Soft Computing Face Recognition\dataset"
CATEGORIES = ["without_mask", "correct_mask", "chin_mask", "mouth_chin_mask", "nose_mouth_mask"]


for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    print(path)
    for subfolder in os.listdir(path): 
        filepath = os.path.join(path, subfolder)
        print(filepath)
        split = re.split('[_.]', subfolder)
        filepathname = subfolder
        
        if len(split) == 5:
            newFolderName = split[1]+split[2]+split[3]
        else:
            print(split[1]+split[2])
            newFolderName = split[1]+split[2]
        print(filepath)
        newPath = os.path.join(DIRECTORY, newFolderName)
        newPath = os.path.join(newPath, subfolder)
        print(newPath)
        shutil.move(filepath, newPath)