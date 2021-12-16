"""
This file is to prepare training and testing txt files which contains path of the images and save it in required path
"""

import os
import glob
import pandas as pd

curr_path = os.getcwd()
trainingTestingFilesPath = curr_path + '/yolov4/darknet/bcd_data/'

"""
Images
"""

images = glob.glob(curr_path + '/yolov4/darknet/bcd_data/bcd_images/*.jpg')
images_lst = list(map(lambda image : 'bcd_data/bcd_images/' + image.split('/')[-1], images))
# print(images_lst)

training_images_lst = images_lst[:int(0.8 * len(images_lst))]
testing_images_lst = images_lst[int(0.8 * len(images_lst)):]

# print(training_images_lst)
# print(testing_images_lst)

training_images_lst_df = pd.DataFrame(training_images_lst)
testing_images_lst_df = pd.DataFrame(training_images_lst)

training_images_lst_df.to_csv(trainingTestingFilesPath+'/bcd_training.txt', header = False, index = False, sep = ' ')
testing_images_lst_df.to_csv(trainingTestingFilesPath+'/bcd_testing.txt', header = False, index = False, sep = ' ')