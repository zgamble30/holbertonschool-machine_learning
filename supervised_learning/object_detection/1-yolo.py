#!/usr/bin/env python3
"""
Yolo Class for object detection using Yolo v3 algorithm
"""

import tensorflow.keras as K
import numpy as np


class Yolo:
    """Yolo class"""

    def __init__(self, model_path, classes_path, class_t, nms_t, anchors):
        """
        Initializes the Yolo object.

        Parameters:
            model_path (str): Path to where a Darknet Keras model is stored.
            classes_path (str): Path to where the list of class names used
                for the Darknet model can be found.
            class_t (float): Box score threshold for the initial
                filtering step.
            nms_t (float): IOU threshold for non-max suppression.
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

    def process_outputs(self, outputs, image_size):
        """
        Process Darknet model outputs.

        Parameters:
            outputs (list): List of numpy.ndarrays containing predictions
                            from the Darknet model for a single image.
            image_size (numpy.ndarray): Array containing the image’s original size
                                        [image_height, image_width].

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
                boxes: List of numpy.ndarrays of shape (grid_height, grid_width, anchor_boxes, 4)
                       containing the processed boundary boxes for each output.
                       (x1, y1, x2, y2) represent the boundary box relative to the original image.
                box_confidences: List of numpy.ndarrays of shape (grid_height, grid_width, anchor_boxes, 1)
                                 containing the box confidences for each output.
                box_class_probs: List of numpy.ndarrays of shape (grid_height, grid_width, anchor_boxes, classes)
                                 containing the box’s class probabilities for each output.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        for output in outputs:
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Compute bounding box coordinates
            t_x, t_y, t_w, t_h = output[:, :, :, :4].T
            grid = np.indices((grid_height, grid_width)).T
            bx = (sigmoid(t_x) + grid[:, :, np.newaxis]) / grid_width
            by = (sigmoid(t_y) + grid[:, :, np.newaxis]) / grid_height
            bw = (anchors[:, :, 0] * np.exp(t_w)) / self.model.input.shape[1].value
            bh = (anchors[:, :, 1] * np.exp(t_h)) / self.model.input.shape[2].value

            # Compute bounding box coordinates (x1, y1, x2, y2)
            x1 = (bx - bw / 2) * image_size[1]
            y1 = (by - bh / 2) * image_size[0]
            x2 = (bx + bw / 2) * image_size[1]
            y2 = (by + bh / 2) * image_size[0]

            boxes.append(np.stack([x1, y1, x2, y2], axis=-1))

            # Box confidence
            box_confidences.append(sigmoid(output[:, :, :, 4:5]))

            # Box class probabilities
            box_class_probs.append(sigmoid(output[:, :, :, 5:]))

        return boxes, box_confidences, box_class_probs

    def sigmoid(self, x):
        """Sigmoid function."""
        return 1 / (1 + np.exp(-x))
