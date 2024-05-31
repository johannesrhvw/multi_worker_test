import logging
import math
from queue import Queue

import cv2
import numpy as np
from configuration.config import GeneralConfig
from object_handling.objects import DetectedObject
from ultralytics.engine.results import Results

from .classification_worker import ClassificationWorker


class ClsfHandler:
    def __init__(self, config: GeneralConfig) -> None:
        """
        Class handling dictionaries containing newly detected objects.

        Args:
            config (GeneralConfig):
            The config containing section for yolo classification inference.
        """
        consumer_config = config.consumer_config
        self.logger = logging.getLogger(__name__)
        self.number_of_workers = consumer_config.yolo_clsfy.number_of_workers
        self.roi_margins = consumer_config.object_roi_filter
        self.input_queues: list[Queue] = []
        self.output_queues: list[Queue] = []
        self.output_shape = (
            consumer_config.yolo_clsfy.input_width,
            consumer_config.yolo_clsfy.input_height,
        )
        self.worker_list: list[ClassificationWorker] = []
        for _ in range(self.number_of_workers):
            input_queue: Queue = Queue()
            output_queue: Queue = Queue()
            self.worker_list.append(ClassificationWorker(config, input_queue, output_queue))
            self.input_queues.append(input_queue)
            self.output_queues.append(output_queue)

    def setup(self) -> None:
        """
        Sets up the classification model, call this function in the new process
        the classification will run in to prevent pickling errors.
        """
        for worker in self.worker_list:
            worker.start()

    def classify(
        self, frame: np.ndarray, detected_objects: dict[str, DetectedObject]
    ) -> dict[str, DetectedObject] | None:
        """
        Adds the classified defects to the detected objects, by running
        yolo-cls inference on the cut out image segments from previous
        yolo-seg results.

        Args:
            frame (np.ndarray): Original frame to crop the objects from.
            detected_objects (dict[str, DetectedObject]): Results from yolo-seg.

        Returns:
            dict[str, DetectedObject] | None: Updated detected objects,
            containing the classified defetcts, if any were detected.
        """
        segments: dict[str, np.ndarray] = {}
        # iterate through all detected objects
        for key, value in detected_objects.items():
            # cut out an image for inference using the bbox and mask
            instance_segment = self._crop_and_mask_image_opencv(frame, value.box, value.mask)
            segments[key] = instance_segment
            # run yolo-cls inference on the cut out image
        batch_size = math.ceil(len(segments) / self.number_of_workers)
        image_count = 0
        batch = []
        batch_dict = {}
        for key, seg in segments.items():
            image_count += 1
            worker_number = image_count // batch_size
            batch_index = image_count % batch_size
            batch.append(seg)
            batch_dict[key] = (worker_number, batch_index)
            if len(batch) == batch_size:
                self.input_queues[worker_number].put(batch)
                batch = []
        batch_results = []
        for queue in self.output_queues:
            item = queue.get()
            batch_results.append(item)
        for key, obj in detected_objects.items():
            batch_info = batch_dict[key]
            worker_number = batch_info[0]
            batch_index = batch_info[1]
            result = batch_results[worker_number][batch_index]
            ret = self._parse_result(result, obj)
            if ret is not None:
                detected_objects[key] = ret
        return detected_objects

    def _parse_result(self, result: Results, detected_object: DetectedObject) -> DetectedObject | None:
        """
        Parses the yolo inference results.

        Args:
            result (Results): Yolo results.
            detected_object (DetectedObject): Object to apply detections to.

        Returns:
            DetectedObject | None: Object updated with the defects classes.
        """
        if result.probs is None:
            self.logger.info(
                f"results.probs = None "
                f"for obj.id={detected_object.id}, "
                f"skipping classification for this object."
            )
            return None
        cls_index = result.probs.top1
        cls_name = result.names[cls_index]
        if cls_name not in detected_object.defects:
            match cls_name:
                case "greening":
                    detected_object.defects.append("greening")
                case "scab":
                    detected_object.defects.append("scab")
                case "cut":
                    detected_object.defects.append("cut")
                case "malformed":
                    detected_object.defects.append("malformed")
                case "bite":
                    detected_object.defects.append("bite")
                case "wire_worm":
                    detected_object.defects.append("wire_worm")
                case "sprout":
                    detected_object.defects.append("sprout")
                case "rot":
                    detected_object.defects.append("rot")
                case _:
                    pass
        return detected_object

    def _crop_and_mask_image_opencv(self, in_image: np.ndarray, in_bbox: np.ndarray, in_mask: np.ndarray) -> np.ndarray:
        """
        Crop out the bounding box and mask an image using the provided polygon.

        Args:
            image (np.ndarray): Image to process.
            bbox (np.ndarray): Normalized bounding box coords to crop.
            polygon (np.ndarray): Normalized polygon coords to mask.

        Returns:
            np.ndarray: segmented image.
        """
        image = in_image.copy()
        polygon = in_mask.copy()
        bbox = in_bbox.copy()
        # Convert bbox to pixel coordinates and crop the image
        target_size = self.output_shape
        height, width = image.shape[:2]
        x1, y1, x2, y2 = (np.array([width, height, width, height]) * bbox).astype(int)
        cropped_image = image[y1:y2, x1:x2]

        # Adjust and scale polygon points to the cropped image size
        polygon[:, 0] = (polygon[:, 0] - bbox[0]) / (bbox[2] - bbox[0]) * (x2 - x1)
        polygon[:, 1] = (polygon[:, 1] - bbox[1]) / (bbox[3] - bbox[1]) * (y2 - y1)
        # Convert contour to signed integer data type and reshape to 1 2
        contour = polygon.astype(np.int32).reshape(-1, 1, 2)
        # Create a binary mask
        b_mask = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        mask = cv2.drawContours(b_mask, [contour], -1, (255, 255, 255), cv2.FILLED)

        # Convert binary mask to a 3 channel image
        ch_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        cropped_image = cropped_image.astype(np.uint8)
        ch_mask = ch_mask.astype(np.uint8)

        # Apply the mask to the cropped image
        masked_image = cv2.bitwise_and(cropped_image, ch_mask)

        # Pad the image to the target size
        delta_width = target_size[0] - masked_image.shape[1]
        delta_height = target_size[1] - masked_image.shape[0]
        top = max(0, delta_height // 2)
        bottom = max(0, delta_height - (delta_height // 2))
        left = max(0, delta_width // 2)
        right = max(0, delta_width - (delta_width // 2))
        return cv2.copyMakeBorder(
            masked_image,
            top,
            bottom,
            left,
            right,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )
