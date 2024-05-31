import logging
import sys
import time

from camera.file_frame_provider import FileFrameProvider
from configuration.config import Config
from configuration.logging_setup import setup_logger
from frame_processing.frame_consumer import FrameConsumer


CONFIG_FILE = "config/config.json"
WORKER_FILE = "data/cam2.avi"


def main_process():
    logger = logging.getLogger(__name__)
    setup_logger("logs/main.log")
    config = Config(CONFIG_FILE)
    cam_config = config.camera_configs[config.serial_numbers[0]]
    worker = FileFrameProvider(WORKER_FILE, cam_config)
    consumer = FrameConsumer(config.camera_configs[config.serial_numbers[0]])
    logger.debug("Setup the consumer.")
    consumer.setup()
    while True:
        frame, frame_id = worker.run()
        _ = consumer.run(frame, frame_id)
        time.sleep(1)


if __name__ == "__main__":
    sys.exit(main_process())
