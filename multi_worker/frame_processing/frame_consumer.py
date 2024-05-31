import logging

import numpy as np
from configuration.config import CameraConfig
from object_handling.objects import DetectedObject

from .classification.clsf_handler import ClsfHandler
from .segmentation.seg_handler import SegHandler


class FrameConsumer:
    def __init__(self, config: CameraConfig) -> None:
        """
        This class runs inference of yolo segmentation
        and classification on a given frame from a camera
        and outputs the classified detected objects.

        Args:
            config (CameraConfig): Configuration of the camera,
            including the settings for segmentation and classification.
        """
        self.logger = logging.getLogger(__name__)
        self.running = False
        self.camera_serial_number = config.serial_number
        self.seg_handler = SegHandler(config)
        self.clsf_handler = ClsfHandler(config.general_config)

    def setup(self) -> None:
        """
        Setup function calling the stup functions of the segmentation
        and classification handlers. This is necessary to be called
        inside the new process before running the inference.
        """
        self.seg_handler.setup()
        self.clsf_handler.setup()

    def run(
        self,
        frame: np.ndarray,
        frame_number: int,
    ) -> dict[str, DetectedObject] | None:
        """
        Segmentation, tracking and classification of objects in a frame.
        A specified ROI is applied to segmented objects too keep only the
        complete objects in the frame.

        Args:
            frame (np.ndarray): Frame for inference.

        Returns:
            dict[str, DetectedObject] | None: Dictionary of classified detected
            objects, the key indicates
            the serialnumber followed by the trackingid.
        """
        # segmentation on the frame
        yolo_result = self.seg_handler.segment(frame)
        if yolo_result is None:
            return None
        tracked_objects = self.seg_handler.track_objects(yolo_result)
        if tracked_objects is None or len(tracked_objects) < 1:
            self.logger.debug("No objects detected.")
            return None
        detected_objects = self.seg_handler.process_norfair_tracking_results(tracked_objects, frame_number)
        classified_objects = self.clsf_handler.classify(frame, detected_objects)
        if classified_objects is None:
            self.logger.debug("No objects classified.")
            return None
        return detected_objects
