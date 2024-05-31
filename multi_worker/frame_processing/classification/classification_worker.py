from queue import Queue
from threading import Thread

import numpy as np
from configuration.config import GeneralConfig

from .classification import Classification


class ClassificationWorker(Thread):
    def __init__(self, config: GeneralConfig, input_queue: Queue, output_queue: Queue) -> None:
        super().__init__()
        self.classifier = Classification(config)
        self.input = input_queue
        self.output = output_queue

    def setup(self) -> None:
        self.classifier.setup()

    def run(self) -> None:
        self.classifier.setup()
        while True:
            item = self.input.get()
            if item is None:
                break
            if isinstance(item, list) and all(isinstance(frame, np.ndarray) for frame in item):
                results = self.classifier.forward(item)
                self.output.put(results)
