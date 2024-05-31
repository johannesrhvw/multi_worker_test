import json

import numpy as np


MAXIMUM_TRACK = 4


class DetectedObject:
    def __init__(
        self,
        obj_id: int,
        serial_number: str,
        detection_data: dict,
    ) -> None:
        """
        Object class containing tracking, detection and segmentation
        result information.

        Args:
            obj_id (int): Tracking id of the object.
            serial_number (str): Serial number of the camera that detected
            the object.
            detection_data (dict): Dictionary containing tracking, detection and
            segmentation results. Keys: mask, box, centroid, cls, conf, age,
            last_distance, velocity, frame.
        """
        # Detection information
        self.mask: np.ndarray = detection_data.get("mask", np.zeros((0, 0)))
        self.box: np.ndarray = detection_data.get("box", np.zeros((0, 0, 0, 0)))
        self.centroid: np.ndarray = detection_data.get("centroid", np.zeros((0, 0)))
        self.class_values: np.ndarray = detection_data.get("cls", np.zeros((0, 0, 0, 0)))
        self.confidence_value: float = detection_data.get("conf", 0.0)
        # tracking information
        self.id: int = obj_id
        self.serial_number: str = serial_number
        self.velocity: np.ndarray = detection_data.get("velocity", [0.0, 0.0])
        self.tracking_age: int = detection_data.get("age", 0)
        self.last_distance: float = detection_data.get("last_distance", 0.0)
        self.frame = detection_data.get("frame", 0)
        self.defects: list[str] = []
        self.defect_probs: list[float] = []

    def json_object_information(self) -> str:
        """
        Converts the object's information to a JSON string.

        Returns:
            str: JSON string containing the object's information.
        """
        # Convert the object's dictionary to a new dictionary
        # Convert numpy arrays to lists
        data = {
            "id": self.id,
            "defects": self.defects,
            "defect_probs": self.defect_probs,
            "centroid": np_array_to_float_list(self.centroid),
            "class": np_array_to_float_list(self.class_values),
            "confidence": round(float(self.confidence_value), 2),
            "tracking_age": self.tracking_age,
            "velocity": np_array_to_float_list(self.velocity),
            "last_distance": round(float(self.last_distance), 2),
            "bounding_box": np_array_to_float_list(self.box),
            "segmentation_mask": np_array_to_float_list(self.mask),
        }
        # Convert the dictionary to a JSON string
        return json.dumps(data)


def np_array_to_float_list(array: np.ndarray) -> list:
    """
    Convert a numpy array to a list of floats.

    Args:
        array (np.ndarray): NumPy array to be converted.

    Returns:
        list: List of floats.
    """
    if array.ndim == 0:
        return [round(float(array), 2)]
    return [np_array_to_float_list(a) for a in array]


class SortingObject:
    def __init__(self, detected_obj: DetectedObject) -> None:
        """
        Object class containing a detected object and its sorting information.

        Args:
            detected_obj (DetectedObject): Detected Object with new detection
            information used to decide if and where the object will be sorted.
        """
        self.id = detected_obj.id
        self.serial_number = detected_obj.serial_number
        self.detected_object = detected_obj
        self.defects = detected_obj.defects
        self.ready_to_sort = False
        self.sorted = False
        self.updated = True
        self.paddle_number = 0
        self.entry_point = detected_obj.centroid[:2]
        self.current_point = detected_obj.centroid[:2]
        self.frame_count = 0
        self.track_count = 0
        self.entry_frame = detected_obj.frame
        self.current_frame = detected_obj.frame

    def update(self, detected_obj: DetectedObject) -> None:
        """
        Update the object with new detection information.

        Args:
            detected_obj (DetectedObject): Detected Object with new detection
            information used to decide if and where the object will be sorted.

        Raises:
            ValueError: When the provided object does not have the
            same id or serialnumber.
        """
        if self.id != detected_obj.id:
            raise ValueError("Object ID mismatch.")
        if self.serial_number != detected_obj.serial_number:
            raise ValueError("Serial number mismatch.")
        self.detected_object = detected_obj
        self.current_point = detected_obj.centroid[:2]
        self.current_frame = detected_obj.frame
        self.updated = True
        # this is updated so that the object is not lost if it is not detected
        if self.track_count < MAXIMUM_TRACK:
            self.track_count += 1
        # update defects for access in the sorting process
        for defect in detected_obj.defects:
            if defect not in self.defects:
                self.defects.append(defect)
