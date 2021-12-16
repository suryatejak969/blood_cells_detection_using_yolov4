"""
This python file is to save our each image in required format which is .jpg and creating annotations file, and saving images
and annotations file in darknet -> bcd_data path

Images and csv file is in current directoy of this file/data
"""

import os
import pandas as pd
import cv2

"""
get the current directory path
"""

curr_path = os.getcwd()

"""
set path for csv file which has each object of our images x_min, y_min, x_max, y_max, image name, label and to images
"""

csvFilePath = curr_path+'/data'
imagesPath = curr_path + '/data/images'
finalImagesAnnotationsPath = curr_path + '/yolov4/darknet/bcd_data/bcd_images'
labelsFolder = curr_path + '/yolov4/darknet/bcd_data/bcd_labels'

"""
read csv file
"""

ann = pd.read_csv(csvFilePath + '/annotations.csv')

# print(ann.head())

"""
Adding for new columns dataframe category, x_center, y_center, width, height    
"""

ann['categoryId'] = ''
ann['x_center'] = ''
ann['y_center'] = ''
ann['width'] = ''
ann['height'] = ''

"""
Setting category 0 to rbc and 1 to wbc
"""

ann.loc[ann['label'] == 'rbc', 'categoryId'] = 0
ann.loc[ann['label'] == 'wbc', 'categoryId'] = 1

"""
Calculation x_center, y_center, width and hegiht from given x_min, y_min, x_max and y_max values
"""

ann['x_center'] = (ann['xmin'] + ann['xmax']) / 2
ann['y_center'] = (ann['ymin'] + ann['ymax']) / 2
ann['width'] = ann['xmax'] - ann['xmin']
ann['height'] = ann['ymax'] - ann['ymin']

# print(ann.head())

"""
Reading each image and it's respective annotations from ann dataframe
"""

os.chdir(imagesPath)

for current_dir, dirs, files in os.walk('.'):
    for file in files:
        if file.endswith('.png'):

            image_png = cv2.imread(file)

            h, w = image_png.shape[:2]

            image_name = file[:-4]

            final_ann = ann.loc[ann['image'] == file].copy()

            final_ann['x_center'] = final_ann['x_center'] / w
            final_ann['y_center'] = final_ann['y_center'] / h
            final_ann['width'] = final_ann['width'] / w
            final_ann['height'] = final_ann['height'] / h

            res_frame = final_ann.loc[:, ['categoryId',
                                         'x_center',
                                          'y_center',
                                          'width',
                                          'height']].copy()
            
            if res_frame.isnull().values.all():
                continue

            """
            Save the annotations in txt file and images in jpg format in our required path which is final images and annotations path
            """

            path_to_save = finalImagesAnnotationsPath

            res_frame.to_csv(path_to_save+'/'+image_name+'.txt', header = False, index = False, sep = ' ')
            res_frame.to_csv(labelsFolder+'/'+image_name+'.txt', header = False, index = False, sep = ' ')

            cv2.imwrite(path_to_save+'/'+image_name+'.jpg', image_png)