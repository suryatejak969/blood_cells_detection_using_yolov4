"""
After training the yolov4 model in google colab, saved the best weight and cofiguration file in current directory of this file
"""

import cv2
from numpy import random
from numpy.lib.type_check import imag
import pandas as pd
import os
import numpy as np
import time

curr_path = os.getcwd()

testingFilePath = curr_path + '/yolov4/darknet/bcd_data/bcd_testing.txt'

with open(testingFilePath) as f:
    lines = f.readlines()

random = np.random.randint(0, len(lines))
print(random)

image = cv2.imread('yolov4/darknet/bcd_data/bcd_images/' + lines[random].replace('\n', '').split('/')[-1])

# image = cv2.resize(image, (416, 416))

# print('/data/images/' + lines[0].replace('\n', '').split('/')[-1])
# print(image.shape)

cv2.namedWindow('Original Image', cv2.WINDOW_NORMAL)
cv2.imshow('Original Image', image)
cv2.waitKey(0)
cv2.destroyWindow('Original Image')

h, w = image.shape[:2]

blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
                             swapRB=True, crop=False)

with open('yolov4/darknet/bcd_data/bcd.names') as f:
    labels = [line.strip() for line in f]

network = cv2.dnn.readNetFromDarknet('bcd_yolov4.cfg',
                                     'bcd_yolov4_best.weights')
layers_names_all = network.getLayerNames()

layers_names_output = [layers_names_all[i - 1] for i in network.getUnconnectedOutLayers()]

probability_minimum = 0.5

threshold = 0.3

colours = np.random.randint(0, 255, size=(len(labels), 3), dtype='uint8')

network.setInput(blob)  # setting blob as input to the network
start = time.time()
output_from_network = network.forward(layers_names_output)
end = time.time()

print('Objects Detection took {:.5f} seconds'.format(end - start))

bounding_boxes = []
confidences = []
class_numbers = []


for result in output_from_network:
    for detected_objects in result:
        scores = detected_objects[5:]
        class_current = np.argmax(scores)
        confidence_current = scores[class_current]

        if confidence_current > probability_minimum:
            box_current = detected_objects[0:4] * np.array([w, h, w, h])

            # print(box_current)

            x_center, y_center, box_width, box_height = box_current
            x_min = int(x_center - (box_width / 2))
            y_min = int(y_center - (box_height / 2))
            print(x_min, y_min, box_width, box_height)

            bounding_boxes.append([x_min, y_min, int(box_width), int(box_height)])
            confidences.append(float(confidence_current))
            class_numbers.append(class_current)

results = cv2.dnn.NMSBoxes(bounding_boxes, confidences,
                           probability_minimum, threshold)

counter = 1

if len(results) > 0:
    for i in results.flatten():
        print('Object {0}: {1}'.format(counter, labels[int(class_numbers[i])]))

        counter += 1

        x_min, y_min = bounding_boxes[i][0], bounding_boxes[i][1]
        box_width, box_height = bounding_boxes[i][2], bounding_boxes[i][3]

        colour_box_current = colours[class_numbers[i]].tolist()

        cv2.rectangle(image, (x_min, y_min),
                      (x_min + box_width, y_min + box_height),
                      colour_box_current, 2)

        if x_min in image.flatten():
            print('true')

        text_box_current = '{}: {:.4f}'.format(labels[int(class_numbers[i])],
                                               confidences[i])

        cv2.putText(image, text_box_current, (x_min, y_min - 5),
                    cv2.FONT_HERSHEY_COMPLEX, 0.4, colour_box_current, 2)


print()
print('Total objects been detected:', len(bounding_boxes))
print('Number of objects left after non-maximum suppression:', counter - 1)

cv2.namedWindow('Detections', cv2.WINDOW_NORMAL)
cv2.imshow('Detections', image)
cv2.waitKey(0)
cv2.destroyWindow('Detections')