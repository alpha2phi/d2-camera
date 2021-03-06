import sys
import cv2 as cv
import logging
import torch
import torchvision

# Log to stdout
logging.basicConfig(stream=sys.stdout, level=logging.INFO)

# GPU is available
gpu = torch.cuda.is_available()
logging.info(f"GPU available - { gpu }")


def object_detection():
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

        # Our operations on the frame come here
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Display the resulting frame
        cv.imshow("frame", gray)
        if cv.waitKey(1) == ord("q"):
            break

    # When everything done, release the capture
    cap.release()
    cv.destroyAllWindows()
