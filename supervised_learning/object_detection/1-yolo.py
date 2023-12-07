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
        Process the predictions from the Darknet model for a single image.

        Parameters:
            outputs (list of numpy.ndarray): Predictions from the Darknet model.
            image_size (numpy.ndarray): Original size of the image.

        Returns:
            tuple: (boxes, box_confidences, box_class_probs)
                boxes: List of numpy.ndarrays containing processed boundary boxes.
                box_confidences: List of numpy.ndarrays containing box confidences.
                box_class_probs: List of numpy.ndarrays containing class probabilities.
        """
        boxes = []
        box_confidences = []
        box_class_probs = []

        # Process each output
        for output in outputs:
            # Extract relevant information from the output
            grid_height, grid_width, anchor_boxes, _ = output.shape

            # Process boundary boxes
            processed_boxes = output[..., :4]
            processed_boxes[..., :2] = 1.0 / (1.0 + np.exp(-processed_boxes[..., :2]))
            processed_boxes[..., 2:] = np.exp(processed_boxes[..., 2:])
            processed_boxes[..., :2] += np.indices((grid_height, grid_width)).T
            processed_boxes[..., :2] /= (grid_width, grid_height)
            processed_boxes[..., 2:] *= self.anchors

            # Convert boundary boxes to (x1, y1, x2, y2) format
            processed_boxes[..., :2] -= processed_boxes[..., 2:] / 2
            processed_boxes[..., 2:] += processed_boxes[..., :2]

            # Scale boundary boxes to the original image size
            processed_boxes[..., 0] *= image_size[1]
            processed_boxes[..., 1] *= image_size[0]
            processed_boxes[..., 2] *= image_size[1]
            processed_boxes[..., 3] *= image_size[0]

            boxes.append(processed_boxes)

            # Process box confidences
            box_confidences.append(1.0 / (1.0 + np.exp(-output[..., 4:5])))

            # Process class probabilities
            box_class_probs.append(1.0 / (1.0 + np.exp(-output[..., 5:])))

        return boxes, box_confidences, box_class_probs
