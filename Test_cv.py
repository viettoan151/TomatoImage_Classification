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
from input_processing import classes

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

if __name__ =='__main__':
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument('--image', type=str, default='YL (18).jpg', metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--use-cuda', action='store_true', default= False)

    # parse argument
    args = parser.parse_args()

    model = cv2.dnn.readNetFromONNX('tomato.onnx')

    image = cv2.imread(args.image)
    pred_type = evaluate_cv2Dnn(image, model)
    print(pred_type)
    cv2.putText(image, pred_type,
                (10, 30),
                cv2.FONT_HERSHEY_COMPLEX,
                1,
                (255, 255, 255),
                2)
    cv2.namedWindow('Predict', cv2.WINDOW_NORMAL)
    cv2.imshow('Predict', image)
    cv2.waitKey(0)