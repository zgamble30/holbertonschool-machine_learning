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

        for output_index, output in enumerate(outputs):
            grid_height, grid_width, anchor_boxes, _ = output.shape
            raw_boundary_box_coords = output[..., :4]
            rawBoxConfidence = output[..., 4:5]
            raw_box_class_probabilities = output[..., 5:]

            # Applying sigmoid activation to the box confidence
            box_confidence_after_sigmoid = 1 / (1 + np.exp(-rawBoxConfidence))
            box_confidences.append(box_confidence_after_sigmoid)

            # Applying sigmoid activation to the class probabilities
            box_class_probs_after_sigmoid = 1 / (1 + np.exp(-raw_box_class_probabilities))
            box_class_probs.append(box_class_probs_after_sigmoid)

            for cell_y in range(grid_height):
                for cell_x in range(grid_width):
                    for anchor_box_index in range(anchor_boxes):
                        anchor_width, anchor_height = self.anchors[output_index][anchor_box_index]
                        tx, ty, tw, th = raw_boundary_box_coords[cell_y, cell_x, anchor_box_index]

                        # Applying sigmoid activation and offsetting by grid cell location
                        boundaryBoxCenter_x = (1 / (1 + np.exp(-tx))) + cell_x
                        boundaryBoxCenter_y = (1 / (1 + np.exp(-ty))) + cell_y

                        # Applying exponential and scaling by anchor dimensions
                        boundary_box_width = anchor_width * np.exp(tw)
                        boundary_box_height = anchor_height * np.exp(th)

                        # Normalizing by grid and model input dimensions
                        boundaryBoxCenter_x /= grid_width
                        boundaryBoxCenter_y /= grid_height
                        boundary_box_width /= int(self.model.input.shape[1])
                        boundary_box_height /= int(self.model.input.shape[2])

                        # Converting to original image scale
                        top_left_x = (boundaryBoxCenter_x - (boundary_box_width / 2)) * image_size[1]
                        top_left_y = (boundaryBoxCenter_y - (boundary_box_height / 2)) * image_size[0]
                        bottom_right_x = (boundaryBoxCenter_x + (boundary_box_width / 2)) * image_size[1]
                        bottom_right_y = (boundaryBoxCenter_y + (boundary_box_height / 2)) * image_size[0]

                        # Storing the processed boundary box coordinates
                        raw_boundary_box_coords[cell_y, cell_x, anchor_box_index] = [
                            top_left_x,
                            top_left_y,
                            bottom_right_x,
                            bottom_right_y,
                        ]

            boxes.append(raw_boundary_box_coords)

        return boxes, box_confidences, box_class_probs

def sigmoid(x):
    """Sigmoid function."""
    return 1 / (1 + np.exp(-x))
