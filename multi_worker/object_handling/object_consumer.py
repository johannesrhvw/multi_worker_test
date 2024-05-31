import logging

import cv2
import numpy as np
from configuration.config import CameraConfig

from object_handling.object_localization import ObjectLocalization
from object_handling.objects import DetectedObject
from object_handling.objects import SortingObject
from object_handling.paddle_lookup import PADDLE_PINS


EXPORT_PATH = "data/objects/"

GET_TIMEOUT = 0.5
SORT_TIME = 10


class ObjectConsumer:
    def __init__(self, config: CameraConfig) -> None:
        """
        ObjectConsumer class to handle the detected objects and create
        a sorting signal for them.

        Args:
            config (Config): Configuration used for ViewTransforming.
        """
        self.logger = logging.getLogger(__name__)
        self.objects: dict[str, SortingObject] = {}
        self.localizers: dict[str, ObjectLocalization] = {}
        self.localizer = ObjectLocalization(config)
        self.config = config
        self.roi_line = config.general_config.sorter_config.sort_line

    def run(
        self,
        new_objects: dict[str, DetectedObject] | None,
        frame: tuple[np.ndarray, int] | None,
    ) -> tuple[list[int], np.ndarray | None]:
        """
        Run function to update the detected objects and create a sorting signal.

        Args:
            new_objects (dict[str, DetectedObject] | None): The new
            DetectedObjects from the queue.
            frames (dict[str, tuple[np.ndarray, int]] | None): Frames in which
            the new objects were detected.

        Returns:
            list[int]: The sorting signal for the detected objects.
        """
        # update the internal objects dictionary
        self.update_objects(new_objects)
        image: np.ndarray | None = None
        if frame is not None:
            image = self.create_image(frame)
        return (self.create_sorter_signal(), image)

    def update_objects(self, new_objects: dict[str, DetectedObject] | None) -> None:
        """
        Updates the own objects attribute with new objects.

        Args:
            new_objects (dict[str, DetectedObject] | None): New objects used to
            update or create the internal objects dictionary.
        """
        for obj in self.objects.values():
            obj.updated = False
        # check if new objects are available
        if new_objects is not None:
            # iterate through all new objects and check if internal
            # objects contain a similar object
            for key in new_objects:
                new_obj = new_objects[key]
                if key in self.objects:
                    self.objects[key].update(new_obj)
                elif new_obj.centroid[1] < self.roi_line:
                    self.objects[key] = SortingObject(new_objects[key])
        # Create a copy of the objects dictionary
        pop_keys = []
        for key, obj in self.objects.items():
            # if the object was not updated decrease its track count
            if not obj.updated:
                obj.track_count -= 1
                if obj.track_count <= 0:
                    pop_keys.append(key)
        for key in pop_keys:
            self.objects.pop(key)
        for obj in self.objects.values():
            self.localizer.localize_paddle(obj)
            if (
                len(obj.defects) > 0
                and obj.detected_object.centroid[1] > self.roi_line
                and obj.sorted is False
                and obj.updated
            ):
                obj.ready_to_sort = True
                obj.track_count = 0

    def create_image(self, frame: tuple[np.ndarray, int]) -> np.ndarray:
        """
        Create an image with all the detected objects and their information.

        Args:
            frames (dict[str, tuple[np.ndarray, int]]): Frames in which the
            objects will be drawn.

        Returns:
            np.ndarray: Edited image.
        """
        color = (0, 255, 0)
        image = frame[0]
        for obj in self.objects.values():
            if obj.updated and not obj.sorted:
                color = (0, 0, 255) if obj.defects else (0, 255, 0)
                mask = obj.detected_object.mask.copy()
                mask = mask * (image.shape[1], image.shape[0])
                mask = mask.astype(np.int32).reshape(-1, 1, 2)
                image = cv2.polylines(image, [mask], True, color, 3)
                info_str = f"{obj.id}: {obj.defects}, {obj.paddle_number}"
                origin = obj.detected_object.box[:2] * image.shape[:2]
                image = cv2.putText(
                    image,
                    info_str,
                    (int(origin[0]), int(origin[1])),
                    cv2.FORMATTER_FMT_DEFAULT,
                    2,
                    color,
                    4,
                )
            continue
        image = cv2.putText(
            image,
            f"Frame: {frame[1]}",
            (10, 50),
            cv2.FORMATTER_FMT_DEFAULT,
            2,
            (0, 0, 0),
            3,
        )
        return cv2.resize(image, None, fx=0.3, fy=0.3, interpolation=cv2.INTER_AREA)

    def create_sorter_signal(self) -> list[int]:
        """
        Create a sorting signal for the detected objects.

        Returns:
            list[int]: Sorting signal.
        """
        # base signals for each available paddle
        signal_len = len(PADDLE_PINS)
        signal: list[int] = [0] * signal_len
        for obj in self.objects.values():
            # check if a object should be sorted
            if obj.paddle_number < signal_len and obj.ready_to_sort and not obj.sorted:
                # self.logger.info(
                #     f"Updating signal for object {key}, "
                #     f"paddle {obj.paddle_number}."
                # )
                # calculate the correct signal count for the paddle
                # we need: camera fps, enty frame number, exit frame number,
                # entry position and the current position
                signal_count = self.localizer.calculate_sorter_count(obj)
                signal[obj.paddle_number] = signal_count
                obj.sorted = True
        return signal
