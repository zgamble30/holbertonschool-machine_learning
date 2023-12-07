#!/usr/bin/env python3
"""
Yolo Class for object detection using Yolo v3 algorithm
"""

import tensorflow.keras as K
import numpy as np


class Yolo:
    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo object.

        Parameters:
            model_path (str): Path to where a Darknet Keras model is stored.
            classes_path (str): Path to where the list of class names used
            for the Darknet model can be found.
            class_t (float): Box score threshold for the initial
            filtering step.nms_t (float): IOU threshold for
            non-max suppression.
            anchors (numpy.ndarray): Array of shape (outputs, anchor_boxes, 2)
            containing all anchor boxes.

        Attributes:
            model (tensorflow.keras.Model): The Darknet Keras model.
            class_names (list): List of class names for the model.
            class_t (float): Box score threshold for the
            initial filtering step.
            nms_t (float): IOU threshold for non-max suppression.
            anchors (numpy.ndarray): Array of anchor boxes.
        """
        self.model = K.models.load_model(model_path)
        self.class_names = self.load_classes(classes_path)
        self.class_t = class_t
        self.nms_t = nms_t
        self.anchors = anchors

    def load_classes(self, file_path):
        """
        Load and return the list of class names from the specified file.

        Parameters:
            file_path (str): Path to the file containing class names.

        Returns:
            list: List of class names.
        """
        with open(file_path, 'r') as f:
            return [line.strip() for line in f]
