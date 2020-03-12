#!/usr/bin/env python3
"""Test_cv.py: testing network by using OpenCV ."""
__author__ = "Viet Toan"
__copyright__ = "Copyright 2019, AI Group"
__license__ = "BSD 3-Clause"
__version__ = "1.0.0"
__email__ = "viettoan151@gmail.com"
__status__ = "Development"


import argparse
import numpy as np
import cv2
import cv2.dnn
#from input_processing import classes
classes = {'brown': 0, 'orange': 1, 'yellow': 2}

def evaluate_cv2Dnn(image, model):
    # resize image
    image = cv2.resize(image, (200, 200))
    # normalization factor
    fact = 255.0/image.max()

    blob = cv2.dnn.blobFromImage(image, scalefactor = fact, ddepth = cv2.CV_32F)
    model.setInput(blob)
    output= model.forward()

    pred = np.argmax(output)

    for key, val in classes.items():
        if val == pred:
            return key
    return 'None'


def grabcut_genmask(image, iter):
    IMG_H, IMG_W = image.shape[:2]
    image = cv2.resize(image, (IMG_W, IMG_H))
    mask = np.zeros(image.shape[:2], np.uint8)
    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    rect = (1, 1, IMG_W, IMG_H)
    cv2.grabCut(image, mask, rect, bgdModel, fgdModel, iter, cv2.GC_INIT_WITH_RECT)
    #mark2 is 0 with cv.GC_PR_BGD and cv.GC_BGD
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    return mask2

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--image', type=str, default='br.jpg', metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--use-cuda', action='store_true', default= False)

    # parse argument
    args = parser.parse_args()

    model = cv2.dnn.readNetFromONNX('tomato.onnx')

    image_raw = cv2.imread(args.image)
    image = cv2.resize(image_raw, (200, 200))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #mask = grabcut_genmask(image, 6)
    #image = image * mask[:, :, np.newaxis]

    pred_type = evaluate_cv2Dnn(image, model)
    print(pred_type)
    cv2.putText(image_raw, pred_type,
                (10, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                2)
    cv2.namedWindow('Predict', cv2.WINDOW_NORMAL)
    cv2.imshow('Predict', image_raw)
    cv2.waitKey(0)