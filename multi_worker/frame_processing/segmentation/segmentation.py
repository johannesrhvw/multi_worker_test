import logging

import numpy as np
import torch
from configuration.config import GeneralConfig
from ultralytics import YOLO
from ultralytics.engine.results import Results


class Segmentation:
    def __init__(self, config: GeneralConfig) -> None:
        """
        Detection class for object detection using YOLO.

        Args:
            config (YOLOSeg): Configuration object for the YOLO model.
        """
        yolo_config = config.consumer_config.yolo_seg
        self.logger = logging.getLogger(__name__)
        self.model: YOLO | None = None
        self.device: torch.device | None = None
        self.model_path = yolo_config.model_path
        self.input_shape = (yolo_config.input_width, yolo_config.input_height)
        self.confidence_threshold = yolo_config.confidence_threshold
        self.iou = yolo_config.iou_threshold
        self.half = yolo_config.half
        self.use_retina = yolo_config.retina_masks

    def setup(self) -> None:
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger.debug(f"Set device to {self.device}")
        self.model = YOLO(model=self.model_path, verbose=False).to(self.device)
        self.logger.debug(f"Loaded YOLO model from {self.model_path}")

    def forward(self, image: np.ndarray) -> list[Results] | None:
        """
        Detect objects in an image using YOLO.

        Args:
            image (np.ndarray): image to detect objects in.

        Returns:
            list[Results]: Yolo results.
        """
        # check if device and model are set up
        if self.device is None:
            self.logger.error("Segmentation: Device is None, device is not set up.")
            raise TypeError("Device is None, device is not set up.")
        if self.model is None:
            self.logger.error("Segmentation: Model is None, model is not set up.")
            raise TypeError("Model is None, model is not set up.")
        results: list[Results] = self.model.predict(
            source=image,
            conf=self.confidence_threshold,
            iou=self.iou,
            imgsz=self.input_shape,
            half=self.half,
            device=self.device,
            retina_masks=self.use_retina,
        )
        if len(results) == 0:
            # self.logger.debug("SEGMENTATION: No objects detected.")
            return None
        return results
