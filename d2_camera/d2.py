import json
import logging
import os
import random
import sys

import cv2 as cv
import detectron2
from detectron2.utils.logger import setup_logger
import numpy as np
import torch
import torchvision
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog, DatasetCatalog

setup_logger()


# Log to stdout
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

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
    # Default video capture
    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Cannot open camera")
        exit()

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # If frame is read correctly ret is True
        if not ret:
            logging.error("Can't receive frame. Exiting ...")
            break

        cv.imshow("frame", frame)

        if cv.waitKey(1) == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()


def object_detection(frame):
    # Our operations on the frame come here
    outputs = predictor(frame)
    logging.debug(outputs["instances"].pred_classes)
    logging.debug(outputs["instances"].pred_boxes)

    # Display the resulting frame
    v = Visualizer(
        frame[:, :, ::-1], MetadataCatalog.get(cfg.DATASETS.TRAIN[0]), scale=1.2
    )
    out = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv.imshow("frame", out.get_image()[:, :, ::-1])

