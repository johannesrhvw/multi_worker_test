import logging

import cv2
import numpy as np
from configuration.config import CameraConfig

from .paddle_lookup import PADDLE_PINS


class PaddleTransformer:
    def __init__(self, config: CameraConfig) -> None:
        """
        This class is used to calculate the position of a object
        in the field of view, given the corresponding source and target
        view. From the position a paddle number for signal creation
        can be derived.#

        Args:
            config (CameraConfig): Configuration holding the source and
            target view information of a camera.
        """
        self.logger = logging.getLogger(__name__)
        view_transform_conf = config.fov_transformer_settings
        # get the views used for perspective transformation
        self.target = view_transform_conf.target_fov.astype(np.float32)
        self.source = view_transform_conf.source_fov.astype(np.float32)
        self.sw = view_transform_conf.source_width
        self.sh = view_transform_conf.source_height
        self.tw = view_transform_conf.target_width
        self.th = view_transform_conf.target_height
        # create opencv perspective transformer using target and source view
        self.m = cv2.getPerspectiveTransform(self.source, self.target)
        self.paddle_width = config.general_config.sorter_config.paddle_width
        self.num_paddles = len(PADDLE_PINS)

    def transform_points(self, points: np.ndarray) -> np.ndarray:
        """
        Uses a opencv PerspectiveTransform matrix to calculate the point from
        source to target coordinates.

        Args:
            points (np.ndarray): Numpy array of shape (2)

        Returns:
            np.ndarray: Point in target coordinates of shape (2)
        """
        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)

    def transform_normalized_point(self, point: np.ndarray) -> np.ndarray:
        """
        Transforms a normalized point of shape (2) with values between 0 and 1
        in the source view into a point in the target view

        Args:
            point (np.ndarray): Normalized point in source view.

        Returns:
            np.ndarray: Point in target view shape (2).
        """
        point = point * np.array([self.sw, self.sh])
        return self.transform_points(point.reshape(1, 2))

    def calculate_paddle(self, point: np.ndarray) -> int:
        """
        Calculates the number of the paddle corresponding to the
        given location.

        Args:
            point (np.ndarray): Input point in target view.

        Returns:
            int: Paddle number id.
        """
        num = int(point[0][0] / self.paddle_width)
        if num < self.num_paddles:
            return num
        return -1

    def calculate_distance_to_exit(self, point: np.ndarray) -> float:
        pos = self.transform_normalized_point(point)
        return self.th - pos[0][1]
