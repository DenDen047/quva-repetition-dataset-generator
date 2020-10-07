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
    width: int = 112,
    height: int = 112
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
        logger.info(f'video info: {vid_info}')

        # load the annotation data
        logger.info(f'load {npy_fpath}')
        count_frames = np.load(npy_fpath)
        logger.info(f'{count_frames}')

        rep_start = 0
        rep_end = count_frames[-1]
        n_rep = len(count_frames)

        # generate period lengths and counts
        period_lengths = []
        counts = []  # [1,1,1,1,1,1,2,2,2,2,2,3,3,3....]
        count = 1
        previous_frame = 0
        for count_frame in count_frames:
            period_length = count_frame - previous_frame
            period_lengths += [period_length] * period_length
            previous_frame = count_frame

            counts += [count] * period_length
            count += 1

        period_lengths = np.asarray(period_lengths)
        counts = np.asarray(counts)

        # update the number of frames
        n_frames = len(period_lengths)

        # crop the image until `n_frames` frame
        imgs = imgs[:n_frames, :]

        # generate periodicities
        periodicities = np.ones((n_frames,))

        # logging
        logger.info(f'imgs: {imgs.shape}')
        logger.info(f'period_lengths: {period_lengths.shape}')
        logger.info(f'periodicities: {periodicities.shape}')
        logger.info(f'counts: {counts.shape}')

        # save
        output_fpath = os.path.join(
            args.output_dir,
            os.path.basename(npy_fpath)[:-4]
        )
        np.savez_compressed(
            output_fpath,
            imgs=imgs,
            period_lengths=period_lengths,
            periodicities=periodicities,
            counts=counts
        )
