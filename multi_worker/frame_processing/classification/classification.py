import logging

import numpy as np
import torch
from configuration.config import GeneralConfig
from ultralytics import YOLO
from ultralytics.engine.results import Results


class Classification:
    def __init__(self, config: GeneralConfig) -> None:
        """
        Instance of Yolo Classification model for defect classification.

        Args:
            config (GeneralConfig):
            Config section containing settings for yolo inference.
        """
        self.logger = logging.getLogger(__name__)
        yolo_config = config.consumer_config.yolo_clsfy
        self.model_path = yolo_config.model_path
        self.model: YOLO | None = None
        self.device: torch.device | None = None
        self.input_shape = (yolo_config.input_width, yolo_config.input_height)
        self.confidence_threshold = yolo_config.confidence_threshold
        self.iou = yolo_config.iou_threshold
        self.half = yolo_config.half
        self.batch_size = yolo_config.batch_size

    def setup(self) -> None:
        """
        Sets the device to cuda if available and loads the model.
        Run this function inside the process inference will happen in
        to prevent pickling errors.
        """
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.logger.debug(f"Set device to {self.device}")
        self.model = YOLO(model=self.model_path, verbose=False).to(self.device)
        self.logger.debug(f"Loaded YOLO model from {self.model_path}")

    def forward(self, images: list[np.ndarray]) -> list[Results] | None:
        """
        Run yolo-cls inference on the given image.

        Args:
            image (np.ndarray): Image for inference.

        Raises:
            TypeError: If the device and model were not setup.

        Returns:
            Results | None: Yolo result for the given image.
        """
        # check if device and model are setup
        if self.device is None:
            self.logger.error("Classification: Device is None, device is not set up.")
            raise TypeError("Device is None, device is not set up.")
        if self.model is None:
            self.logger.error("Classification: Model is None, model is not set up.")
            raise TypeError("Model is None, model is not set up.")
        results: list[Results] = self.model.predict(source=images, device=self.device, iou=self.iou, half=self.half)
        if results is None or len(results) == 0:
            self.logger.warning("Detected object can not be classified.")
            return None
        return results
