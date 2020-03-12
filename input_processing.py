#!/usr/bin/env python3
"""input_processing.py: processing input folder."""
__author__ = "Viet Toan"
__copyright__ = "Copyright 2019, AI Group"
__license__ = "BSD 3-Clause"
__version__ = "1.0.0"
__email__ = "viettoan151@gmail.com"
__status__ = "Development"

import glob
import cv2
import os
import pandas as pd

# Read image data
# Red tomato
is_training = True

classes = {'brown': 0, 'orange': 1, 'yellow': 2}

if is_training:
    data_folder = 'training\\'
else:
    data_folder = 'testing\\'

def process_input(class_dict, data_folder, resize_flag = True, size = 200, hsv_flag = True):
    dataset_labels = []
    for id_cls, cls in enumerate(list(class_dict.keys())):
        cls_folder = cls + '\\'
        list_files = glob.glob(data_folder + cls_folder + '*.jpg')
        data = []
        for img_id, im_pth in enumerate(list_files):
            img_name = im_pth.split('\\')[-2] + '_' + im_pth.split('\\')[-1].split('.')[0]
            if resize_flag:
                # write to label file
                image = cv2.imread(im_pth)
                # Resize image
                img_X = int(size)
                #img_Y = int(image.shape[0] * (img_X / image.shape[1]))
                img_Y = img_X
                imgresize = cv2.resize(image, (img_X, img_Y))
                # Change to HSV color and extract histogram feature
                if (hsv_flag):
                    image_hsv = cv2.cvtColor(imgresize, cv2.COLOR_BGR2HSV)
                else:
                    image_hsv = imgresize

            # feature folder if necessary
            feature_pth = data_folder + ('feature\\' if resize_flag else '')
            if not os.path.exists(feature_pth):
                os.makedirs(feature_pth)
            # class folder
            label_pth = feature_pth + cls + '\\'
            if not os.path.exists(label_pth):
                os.makedirs(label_pth)

            # image file name
            file_pth = label_pth + img_name + '.jpg'
            if resize_flag:
                cv2.imwrite(file_pth, image_hsv)

            dataset_labels.append([file_pth, str(classes[cls])])
    pd.DataFrame(dataset_labels).to_csv(feature_pth + 'label.csv')

if __name__ == '__main__':
    process_input(classes, data_folder, resize_flag = True, hsv_flag= True)