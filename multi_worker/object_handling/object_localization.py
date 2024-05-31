import logging

import numpy as np
from configuration.config import CameraConfig

from object_handling.objects import SortingObject
from object_handling.paddle_transformer import PaddleTransformer


class ObjectLocalization:
    def __init__(self, config: CameraConfig) -> None:
        """
        Class holding the different view paddle transformers for the
        corresponding cameras.

        Args:
            config (Config): Configuration object containing the view
            transformer settings for each camera.
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.roi_line = config.general_config.sorter_config.sort_line
        self.consumer_delay = config.general_config.consumer_config.consumer_interval
        self.transformer: PaddleTransformer = PaddleTransformer(config)

    def localize_paddle(self, obj: SortingObject) -> None:
        """
        Localizes the object to a paddle.

        Args:
            obj (SortingObject): Object to be sorted.
        """
        position = self._calculate_position(obj)
        # this is a clusterfuck but it works
        # get the transformer with the same serial number as the object
        # use it to calculate the position of the object
        obj.paddle_number = self.transformer.calculate_paddle(position)

    def _calculate_speed(self, obj: SortingObject) -> float:
        """
        Calculate the speed of the given object.

        Args:
            obj (SortingObject): Object to estimate the speed for.

        Returns:
            float: Speed of the object in cm/s.
        """
        transformer = self.transformer
        # clalculate entry position in centimeters
        entry_pos_cm = transformer.transform_normalized_point(obj.entry_point[:2])
        # get entry frame
        entry_frame = obj.entry_frame
        # calculate current position
        current_pos_cm = transformer.transform_normalized_point(obj.current_point[:2])
        # get current frame
        current_frame = obj.current_frame
        # get fps
        fps = self.config.device_settings.fps
        # calculate the speed of the object using all the given parameters
        # delta distance
        distance = current_pos_cm[0][1] - entry_pos_cm[0][1]
        # delta time
        travel_time = (current_frame - entry_frame) / fps
        # speed in cm/s
        return distance / travel_time

    def calculate_sorter_count(self, obj: SortingObject) -> int:
        """
        Calculate the count of sorter iterations needed to reach the sorter.

        Args:
            obj (SortingObject): Object to be sorted.

        Returns:
            int: Interval count until object has to be sorted.
        """
        # get the sorter interval needed to calculate the sorter count
        sorter_interval = self.config.general_config.sorter_config.sort_interval
        # get speed of the object
        speed = self._calculate_speed(obj)
        # get the distance of the object to the exit/sorter
        distance_to_exit = self.transformer.calculate_distance_to_exit(obj.current_point[:2])
        # current iteration delay
        # get the time the object will need to reach the sorter
        time_to_exit = (distance_to_exit / speed) - self.consumer_delay
        if time_to_exit < 0:
            self.logger.error("Delay of object is too high, cannot sort in time.")
            return 0
        # divide by the sorters interval to get value of iterations to be
        # counted down until sorting
        return int(time_to_exit // sorter_interval)

    def _calculate_position(self, obj: SortingObject) -> np.ndarray:
        """
        Calculate the position of the object in the paddle coordinate system.

        Args:
            obj (SortingObject): Object to be sorted.

        Returns:
            np.ndarray: Position of the object in the paddle coordinate system.
        """
        return self.transformer.transform_normalized_point(obj.detected_object.centroid[:2])
