import json
import logging
import multiprocessing as mp
from multiprocessing import Process, Queue
import os
import random
import sys
import time

import cv2
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.engine import DefaultPredictor
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
import numpy as np
import torch
import torchvision

setup_logger()

# Log to stdout
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# Number of processors
logging.info(f"Number of processors: { mp.cpu_count() } ")

# GPU is available
gpu = torch.cuda.is_available()
logging.info(f"GPU available - { gpu }")

cfg = get_cfg()

# add project-specific config (e.g., TensorMask) here if you're not running a model in detectron2's core library
cfg.merge_from_file(
    model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")
)

# set threshold for this model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5

# Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
    "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
)

if not gpu:
    cfg.MODEL.DEVICE = "cpu"

predictor = DefaultPredictor(cfg)


def video_capture():
    """
    Video capture.
    """
    # Default video capture
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Cannot open camera")
        exit()

    input_q = Queue()
    output_q = Queue()
    process = Process(target=object_detection, args=(input_q, output_q))
    process.start()
    processed_frame = None

    cv2.namedWindow("main")
    cv2.namedWindow("object")

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            logging.error("Can't receive frame. Exiting ...")
            break

        if input_q.empty():
            input_q.put(frame)

        concat_frame = frame
        if not output_q.empty():
            processed_frame = output_q.get()
            cv2.imshow("object", processed_frame)

        cv2.imshow("main", frame)

        if cv2.waitKey(1) == ord("q"):
            input_q.close()
            output_q.close()
            input_q.join_thread()
            output_q.join_thread()
            process.terminate()
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


def object_detection(input_q, output_q):
    while True:
        if input_q.empty():
            continue

        frame = input_q.get()

        # Our operations on the frame come here
        outputs = predictor(frame)
        logging.debug(outputs["instances"].pred_classes)
        logging.debug(outputs["instances"].pred_boxes)

        # Display the resulting frame
        v = Visualizer(
            frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2
        )
        out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        output_q.put(out.get_image()[:, :, ::-1])
