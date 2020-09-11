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

    os.makedirs(args.output_dir, exist_ok=True)

    logger.info(f'command line arguments: {args}')

    vid_fpaths = sorted(glob.glob(os.path.join(video_dir, '*.mp4')))
    npy_fpaths = sorted(glob.glob(os.path.join(npy_dir, '*.npy')))

    for vid_fpath, npy_fpath in zip(vid_fpaths, npy_fpaths):
        # load a video
        logger.info(f'load {vid_fpath}')
        imgs, vid_info = read_video(vid_fpath)
        n_frames, w, h, c = imgs.shape
        logger.info(f'{imgs.shape}')
        logger.info(f'{vid_info}')

        # load the annotation data
        logger.info(f'load {npy_fpath}')
        count_frames = np.load(npy_fpath)
        logger.info(f'{count_frames}')

        rep_start = 0
        rep_end = count_frames[-1]
        n_rep = len(count_frames)

        # generate the data for RepNet
        period_length = (rep_end - rep_start) / n_rep
        periodicities = np.ones((n_frames,))

        # save
        output_fpath = os.path.join(
            args.output_dir,
            os.path.basename(npy_fpath)[:-4]
        )
        np.savez_compressed(
            output_fpath,
            imgs, period_length, periodicities
        )
