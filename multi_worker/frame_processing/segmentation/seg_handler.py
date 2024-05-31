import logging

import cv2
import numpy as np
import torch
from configuration.config import CameraConfig
from norfair import Tracker
from norfair.tracker import Detection
from norfair.tracker import TrackedObject
from object_handling.objects import DetectedObject
from torch import Tensor
from ultralytics.engine.results import Boxes
from ultralytics.engine.results import Results

from .segmentation import Segmentation


class SegHandler:
    def __init__(self, config: CameraConfig) -> None:
        """
        Segmentation and tracking manager class for tracking objects
        in consecutive frames.

        Args:
            config (CameraConfig): Configuration of the camera source
            of the frames, providing the serial number and
            settings for segmentation and tracking.
        """
        self.serial_number = config.serial_number
        general_config = config.general_config
        self.logger = logging.getLogger(__name__)
        track_config = general_config.consumer_config.norfair_tracker
        self.tracker = Tracker(
            distance_function=track_config.distance_function,
            distance_threshold=track_config.distance_threshold,
            initialization_delay=track_config.initialization_delay,
            hit_counter_max=track_config.hit_counter_max,
            pointwise_hit_counter_max=track_config.pointwise_hit_counter_max,
            detection_threshold=track_config.detection_threshold,
            past_detections_length=track_config.past_detections_length,
        )
        self.roi_margins = general_config.consumer_config.object_roi_filter
        self.segmentation = Segmentation(general_config)
        self.track_points = track_config.track_points
        self.forward_shape = (
            config.general_config.consumer_config.yolo_seg.input_width,
            config.general_config.consumer_config.yolo_seg.input_height,
        )

        self.device: torch.device | None = None

    def setup(self) -> None:
        """
        Setup function calling the setup functions of the segmentation handler.
        Setting the device to cuda if possible. Run this function inside
        the new Process to prevent pickling errors.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.segmentation.setup()

    def segment(
        self,
        image: np.ndarray,
    ) -> Results | None:
        """
        Segment and track objects in an image.

        Args:
            image (np.ndarray): Image to process.
            recent_objects (dict[int, DetectedObject]): Last tracked objects.

        Returns:
            Results | None: Yolo results of the segmentation.
        """
        resized_image = cv2.resize(image.copy(), self.forward_shape, interpolation=cv2.INTER_AREA)
        yolo_results = self.segmentation.forward(resized_image)
        if yolo_results is None:
            return None
        yolo_result = yolo_results[0]
        # filter results by roi defined in config file
        return self._filter_results(yolo_result)

    def track_objects(
        self,
        yolo_result: Results,
    ) -> list[TrackedObject] | None:
        """
        Update the norfair tracked objects with new yolo detections.

        Args:
            image (np.ndarray): Image to process.
            yolo_results (list[Results] | None):
            Yolo results to apply and track.

        Returns:
            list[TrackedObject]: New and updated norfair tracked objects.
        """
        # Check for any available results
        if yolo_result is None:
            return None
        # filter results by roi defined in config file
        norfair_detections = self._yolo_detections_to_norfair_detections(yolo_result)
        if norfair_detections is None:
            self.logger.debug("Could not convert solo Results to Norfair Detections.")
            return None
        # Update the tracker with the new detections
        return self.tracker.update(detections=norfair_detections)

    def _filter_results(self, result: Results) -> Results | None:
        """
        Filter the results by the ROI defined in the config file.

        Args:
            result (Results): Yolo results to filter.

        Returns:
            Results | None: Yolo results that passed the ROI check.
        """
        if self.device is None:
            self.logger.error("SegHandler: Device is None, device is not set up.")
            raise TypeError("Device is None, device is not set up.")
        if result.boxes is None or result.masks is None:
            return None
        # generate the boolean mask for the boxes that passed the ROI check
        roi_passed_mask = self._check_roi_batch(result.boxes).to(self.device)
        # all tensors must be on the same device
        boxes = result.boxes.to(self.device)
        masks = result.masks.to(self.device)
        # Filter boxes and masks that passed the ROI check
        filtered_boxes = boxes[roi_passed_mask]
        filtered_masks = masks[roi_passed_mask]
        # Only add to filtered_results if there
        # are boxes that passed the ROI check
        if filtered_boxes.shape[0] > 0:
            result.boxes = filtered_boxes
            result.masks = filtered_masks
            return result
        return None

    def _check_roi_batch(self, boxes: Boxes) -> Tensor:
        """
        Checks if the boxes are within the ROI margins.

        Args:
            boxes (Boxes): Ultralytics Boxes object containing
            the bounding boxes from the object detection.

        Returns:
            Tensor: Tensor with booleans indicating if the boxes
            are within the ROI.
        """
        if self.device is None:
            self.logger.error("SegImageProcessor: Device is None,device is not set up.")
            raise TypeError("Device is None, device is not set up.")
        roi_x1 = self.roi_margins.margin_left
        roi_y1 = self.roi_margins.margin_top
        roi_x2 = self.roi_margins.margin_right
        roi_y2 = self.roi_margins.margin_bottom
        within_roi = (
            (roi_x1 <= boxes.xyxyn[:, 0])
            & (boxes.xyxyn[:, 0] <= roi_x2)
            & (roi_y1 <= boxes.xyxyn[:, 1])
            & (boxes.xyxyn[:, 1] <= roi_y2)
            & (roi_x1 <= boxes.xyxyn[:, 2])
            & (boxes.xyxyn[:, 2] <= roi_x2)
            & (roi_y1 <= boxes.xyxyn[:, 3])
            & (boxes.xyxyn[:, 3] <= roi_y2)
        )
        if isinstance(within_roi, Tensor):
            return within_roi
        return torch.tensor(within_roi, device=self.device)

    def _yolo_detections_to_norfair_detections(self, result: Results | None) -> list[Detection] | None:
        """
        Converts yolo results to norfair detections.

        Args:
            result (Results | None): yolo results object.

        Returns:
            list[Detection] | None: List of norfair detections.
        """
        if result is None or result.boxes is None or result.masks is None:
            return None
        # make shure all values are on the cpu and in numpy ndarray format
        # test = result.tojson()
        boxes = result.boxes.xyxyn.cpu().numpy() if isinstance(result.boxes.xyxyn, Tensor) else result.boxes.xyxyn

        centroids = result.boxes.xywhn.cpu().numpy() if isinstance(result.boxes.xywhn, Tensor) else result.boxes.xywhn

        masks = result.masks.xyn.cpu().numpy() if isinstance(result.masks.xyn, Tensor) else result.masks.xyn

        classes = result.boxes.cls.cpu().numpy() if isinstance(result.boxes.cls, Tensor) else result.boxes.cls

        confidences = result.boxes.conf.cpu().numpy() if isinstance(result.boxes.conf, Tensor) else result.boxes.conf

        # create norfair detections from the yolo results detection infos
        if self.track_points == "euclidean":
            norfair_detections = [
                Detection(
                    points=centroid,
                    data={
                        "box": box,
                        "mask": mask,
                        "centroid": centroid,
                        "cls": cls,
                        "conf": conf,
                    },
                )
                for box, mask, centroid, cls, conf in zip(boxes, masks, centroids, classes, confidences, strict=False)
            ]
        elif self.track_points == "bbox":
            norfair_detections = [
                Detection(
                    points=box,
                    data={
                        "box": box,
                        "mask": mask,
                        "centroid": centroid,
                        "cls": cls.item(),
                        "conf": conf.item(),
                    },
                )
                for box, mask, centroid, cls, conf in zip(boxes, masks, centroids, classes, confidences, strict=False)
            ]
        return norfair_detections

    def process_norfair_tracking_results(
        self, tracked_objects: list[TrackedObject], frame_number: int
    ) -> dict[str, DetectedObject]:
        """
        Parses norfairs TrackedObject objects into a dictionary
        of DetectedObjects.

        Args:
            tracked_objects (list[TrackedObject]): Norfairs tracked objects
            to be converted.

        Raises:
            ValueError: If a given TrackedObject has no id associated with it.

        Returns:
            dict[str, DetectedObject]: Dictionary of detected objects, the key
            represents the serial number of the camera that took a image of the
            object and the tracking id provided by norfair.
        """
        objects: dict[str, DetectedObject] = {}
        for track_object in tracked_objects:
            # retrieve the detection data from the tracked object
            data: dict = track_object.last_detection.data
            data["velocity"] = track_object.estimate_velocity
            data["age"] = track_object.age
            data["last_distance"] = track_object.last_distance
            data["frame"] = frame_number
            if track_object.id is None:
                raise ValueError("Instance id of tracked object is None.")
            instance_id = track_object.id
            instance_key = f"{self.serial_number}_{track_object.id}"
            # append a new detected object to the dictionary
            objects[instance_key] = DetectedObject(instance_id, self.serial_number, data)
        return objects
