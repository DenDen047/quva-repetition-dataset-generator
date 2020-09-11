import os
import sys
import csv
import time
import pathlib
import argparse
import logging
from tqdm import tqdm
import cv2 as cv
import numpy as np
from pprint import pprint

import youtube_dl


# setting for argparse
parser = argparse.ArgumentParser(description='Analysis an image showing company structure')
parser.add_argument('--dataset_dir', type=str, default='/data', help='the directory path where the countix csv files and outputs are')
parser.add_argument('--reshape_video', action='store_true', default=False, help='to save the space, the downloaded videos are reshaped into 224x224')
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


def download_video_from_url(
    url_to_video: str,
    path_to_video: str,
    skip_existing_videos: bool = True
) -> str:
    """ This function is copied from https://colab.research.google.com/github/google-research/google-research/blob/master/repnet/repnet_colab.ipynb """

    if os.path.exists(path_to_video):
        if skip_existing_videos:
            return 'skipped'
        else:
            os.remove(path_to_video)
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/mp4',
        'outtmpl': str(path_to_video),
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url_to_video])
        return 'success'


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
    data_dir = pathlib.Path(args.dataset_dir).resolve()
    video_dir = os.path.join(data_dir, 'downloaded_videos')
    npz_dir = os.path.join(data_dir, 'video_frame_data')
    csv_file_names = [
        'countix_train.csv',
        'countix_val.csv',
        'countix_test.csv'
    ]

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)
    os.makedirs(npz_dir, exist_ok=True)

    logger.info(f'command line arguments: {args}')

    for csv_file_name in csv_file_names:
        logger.info(f'load {csv_file_name}')

        with open(os.path.join(data_dir, csv_file_name)) as csv_file:
            csv_reader = csv.DictReader(csv_file)

            for row in csv_reader:
                url = 'http://youtube.com/watch?v={}'.format(row['video_id'])
                logger.debug(url)

                # check if the video was already downloaded
                video_fpath = os.path.join(video_dir, f"{row['video_id']}.mp4")
                if os.path.exists(video_fpath):
                    continue

                # download
                try:
                    r = download_video_from_url(
                        url_to_video=url,
                        path_to_video=video_fpath
                    )
                except Exception as e:
                    logger.debug(e)

                # reshape the video
                if args.reshape_video:
                    imgs, vid_info = read_video(video_fpath)
                    n_frames, w, h, c = imgs.shape

                    def t2f(time):  # time to frame
                        return time * n_frames / vid_info['time']

                    rep_start = float(row['repetition_start'])
                    rep_end = float(row['repetition_end'])
                    n_rep = float(row['count'])

                    period_length = t2f(rep_end - rep_start) / n_rep

                    periodicities = np.zeros((n_frames,))
                    periodicities[
                        round(t2f(rep_start)):round(t2f(rep_end))+1
                    ] = 1

                    # save data & remove the video
                    np.savez_compressed(
                        os.path.join(npz_dir, row['video_id']),
                        imgs, periodicities, period_length
                    )
                    os.remove(video_fpath)
