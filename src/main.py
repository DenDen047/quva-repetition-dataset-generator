import os
import sys
import csv
import time
import pathlib
import argparse
import logging
import glob
from tqdm import tqdm
import cv2 as cv
import numpy as np
from pprint import pprint


# setting for argparse
parser = argparse.ArgumentParser(description='Analysis an image showing company structure')
parser.add_argument('--quva_data_dir', type=str, default='/data/QUVARepetitionDataset', help='the directory path where the QUVA Repetition Dataset are put')
parser.add_argument('--output_dir', type=str, default='/data/output', help='the directory path where the outputs are put')
parser.add_argument('--logging', action='store_true', default=False, help='if true, output the all image/pdf files during the process')

args = parser.parse_args()
log_dir = '/logs/{}'.format(time.strftime("%Y%m%d-%H%M%S"))

# setting for logging
if args.logging:
    log_fpath = os.path.join(log_dir, 'logger.log')
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=log_fpath,
        level=logging.DEBUG
    )
logger = logging.getLogger(__name__)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setLevel(logging.DEBUG)
logger.addHandler(_console_handler)


def read_video(
    video_filename: str,
    width: int = 224,
    height: int = 224
):
    """Read video from file."""
    cap = cv.VideoCapture(video_filename)
    info = {
        'fps': cap.get(cv.CAP_PROP_FPS),
        'n_frame': cap.get(cv.CAP_PROP_FRAME_COUNT),
    }
    frames = []
    if cap.isOpened():
        while True:
            success, frame_bgr = cap.read()
            if not success:
                break
            frame_rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
            frame_rgb = cv.resize(frame_rgb, (width, height))
            frames.append(frame_rgb)
            info['time'] = cap.get(cv.CAP_PROP_POS_MSEC) / 1000.0  # seconds
    frames = np.asarray(frames)
    return frames, info


if __name__ == "__main__":
    quva_dir = pathlib.Path(args.quva_data_dir).resolve()
    video_dir = os.path.join(quva_dir, 'videos')
    npy_dir = os.path.join(quva_dir, 'annotations')

    logger.info(f'command line arguments: {args}')

    npy_fpaths = glob.glob(os.path.join(npy_dir, '*.npy'))

    for npy_fpath in npy_fpaths:
        pprint(npy_fpath)
        result = np.load(npy_fpath)
        pprint(result)

        break