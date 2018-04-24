##############################################################
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################

"""Utilities to read video files. Replicating functionality from
caffe2/video/video_io.cc into python."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import h5py
import os

from utils.general import deprecated
from core.config import cfg

import logging
logger = logging.getLogger(__name__)

# This is important as otherwise OpenCL (compiled with OpenCV) would allocate
# a small amount of memory on GPU 0, and each subprocess I spawn for dataloading
# will do the same. For 64 read threads, it will end up allocating ~4.5G on GPU0
# Was debugged in
# https://www.facebook.com/groups/184236721951559/permalink/468759586832603/
try:
  cv2.ocl.setUseOpenCL(False)
except AttributeError:
  # no ocl in cv2
  pass


def _get_frame_from_capture(cap, frame_number, filename, total_frames):
    frame_pos_const = cv2.cv.CV_CAP_PROP_POS_FRAMES if \
        cv2.__version__.startswith('2') else cv2.CAP_PROP_POS_FRAMES
    if (cap.get(frame_pos_const) != int(frame_number)):
        # Following is an expensive operation; avoid if possible
        cap.set(frame_pos_const, int(frame_number))
    success, I = cap.read()
    if not success:
        logger.error('Unable to read frame {} (out-of {}) from {}. Using 0s.'
                     .format(frame_number, total_frames, filename))
    elif I.max() > 256 or I.min() < 0:
        logger.error('Invalid values detected when reading frame {} '
                     'from {}. Using 0s.'.format(frame_number, filename))
    else:
        return I
    return cfg.PIXEL_MEANS


class ReadVideo():
    def __init__(self, filename, from_everstore):
        self.filename = filename
        self.from_everstore = from_everstore
        # Check if this is an everstore format filename: EV/<handle>
        # TODO(rgirdhar): make 'EV' a constant
        if os.path.dirname(filename) == 'EV':
            self.filename = os.path.basename(filename)
            self.from_everstore = True

    def __enter__(self):
        if not self.from_everstore:
            self.cap = cv2.VideoCapture(self.filename)
        else:
            blobs = read_from_everstore([self.filename], download=False)
            self.cap = None
            if len(blobs) == 0:
                logger.error('Unable to read everstore for {}. Return None.'
                             .format(self.filename))
            else:
                try:
                    self.cap = cv2.VideoCapture(blobs[0])
                except Exception as e:
                    logger.error('Unable to read blob URL {} corr to handle '
                                 '{} as a video due to {}. Returning None.'
                                 .format(blobs[0], self.filename, e))
        return self.cap

    def __exit__(self, type, value, traceback):
        if self.cap.isOpened():
            self.cap.release()


# # Now the frame ids to read are being defined at the ROIdb stage
# # (json_dataset.py). This way I can ensure I have pose outputs for the correct
# # frames.
# def get_frame_ids(key_frame, total_frames=9999999999, sampling_rate=None):
#     nframes = cfg.VIDEO.NUM_FRAMES
#     if sampling_rate is None:
#         sampling_rate = cfg.VIDEO.TIME_INTERVAL
#     key_frame_pos = cfg.VIDEO.NUM_FRAMES // 2  # pos in the output
#     if sampling_rate == 0:
#         # The range formula doesn't work for this corner case. Ideally I would
#         # want to repeat the frame nframes number of times (debugging expts)
#         frames_to_read = [key_frame] * nframes
#     else:
#         frames_to_read = [el % total_frames for el in range(
#             int(key_frame - sampling_rate * key_frame_pos),
#             int(key_frame + sampling_rate * (nframes - key_frame_pos)),
#             int(sampling_rate))]
#     return np.array(frames_to_read)


class WriteVideo():
    def __init__(self, filename, im_shape, fps=24):
        self.filename = filename
        self.fps = fps
        self.shape = im_shape
        assert self.filename.endswith('.avi'), \
            '.avi is only supported writable format'

    def __enter__(self):
        fourcc_func = cv2.cv.CV_FOURCC if cv2.__version__.startswith('2') else \
            cv2.VideoWriter_fourcc
        fourcc = fourcc_func(
            chr(ord('M')),
            chr(ord('J')),
            chr(ord('P')),
            chr(ord('G')))
        self.writer = cv2.VideoWriter(
            self.filename,
            fourcc,
            self.fps, (self.shape[1], self.shape[0]))
        return self.writer

    def __exit__(self, type, value, traceback):
        if self.writer.isOpened():
            self.writer.release()


def WriteClipToFile(filename, frames, fps=24):
    with WriteVideo(filename, frames.shape[1:3], fps) as writer:
        for i in range(frames.shape[0]):
            writer.write(frames[i].astype('uint8'))


def ReadClipFromVideo(
    filename,  # string
    # list of frames, 0-idxed. typically gen using get_frame_ids
    frame_list=np.zeros((0,)),
    from_everstore=False,  # set True if the filename is a handle to everstore
    width=None, height=None,  # Use these to return 0s matrix when can't read
):
    """Read a bunch of frames from a video path.
    Reference for cv2 flags is here: https://stackoverflow.com/a/14776701
    Args:
        filename (str): Path to the video to read. Can be .avi/.mp4 or even can
            be a format string like /path/to/image_%05d.jpg to read from a
            directory with frames
    Returns:
        A NHWC format array with the frames requested
    """
    nframes = len(frame_list)
    frame_list = np.array(frame_list)
    with ReadVideo(filename, from_everstore) as cap:
        if not cap.isOpened():
            logger.error('Unable to read input filename {}.'.format(filename))
            if height is not None and width is not None:
                logger.error('Returning 0s.')
                return np.zeros((nframes, height, width, 3))
            return None

        frame_count = int(cap.get(
            cv2.cv.CV_CAP_PROP_FRAME_COUNT if cv2.__version__.startswith('2')
            else cv2.CAP_PROP_FRAME_COUNT))
        wd = int(cap.get(
            cv2.cv.CV_CAP_PROP_FRAME_WIDTH if cv2.__version__.startswith('2')
            else cv2.CAP_PROP_FRAME_WIDTH))
        ht = int(cap.get(
            cv2.cv.CV_CAP_PROP_FRAME_HEIGHT if cv2.__version__.startswith('2')
            else cv2.CAP_PROP_FRAME_HEIGHT))
        if np.any(frame_list >= frame_count):
            logger.error('Got a frame list with elements outside the video! '
                         'Got {frame_list}, video total frames {frame_count} '
                         'for filename {filename}. Modulo-ing each element of the '
                         'list.'.format(frame_list=frame_list,
                                        frame_count=frame_count,
                                        filename=filename))
            frame_list = frame_list % frame_count

        res = np.zeros((nframes, ht, wd, 3))  # allocate memory to store outputs
        res[:, ...] = cfg.PIXEL_MEANS  # in case I choose to leave some images blank

        # Zip it with the position where it should be copied to.
        frames_to_copy = zip(range(nframes), frame_list.tolist())
        # start decoding
        for target_pos, frame_number in frames_to_copy:
            res[target_pos, ...] = _get_frame_from_capture(
                cap, frame_number, filename, frame_count)
    return res


def StackIntoChannels(vid):
    """ Convert a video in NxHxWxC format to HxWx(C*N) """
    return np.concatenate(np.split(vid, vid.shape[0], axis=0), axis=-1)[0]


def UnstackFromChannels(vid):
    """ Convert a video in HxWx(C*N) format to NxHxWxC """
    return np.stack(np.split(vid, vid.shape[-1] / 3, axis=-1), axis=0)


@deprecated
def VideoToStringArray(video_array):
    """Converts a NCHW video array to a N length string array with
    JPEG encoded strings, to be able to store as h5 files.
    """
    nframes = video_array.shape[0]
    frames = np.split(np.transpose(video_array, (0, 2, 3, 1)), nframes, axis=0)
    # np.void from http://docs.h5py.org/en/latest/strings.html
    frames = np.array([np.void(cv2.imencode(
        '.jpg', frame[0])[1].tostring()) for frame in frames])
    return frames


def VideoToH5(video_array, outpath):
    """Stores a NCHW video array to a N length H5 file.
    """
    nframes = video_array.shape[0]
    frames = np.split(np.transpose(video_array, (0, 2, 3, 1)), nframes, axis=0)
    frames = [cv2.imencode('.jpg', frame[0])[1] for frame in frames]
    with h5py.File(outpath, 'w') as fout:
        grp = fout.create_group('frames')
        for i in range(len(frames)):
            grp.create_dataset(str(i), data=frames[i].astype(np.uint8))


def H5ToVideo(fpath, frames_to_read=None):
    """Read frames from the h5 file.
    Args:
        fpath (str): Filename of h5 file
        frames_to_read (list or None): List of frames to read. None => all frames
    """
    res = []
    with h5py.File(fpath, 'r') as fin:
        nframes = len(fin['frames'])
        if frames_to_read is None:
            frames_to_read = range(nframes)
        grp = fin['frames']
        for fid in frames_to_read:
            res.append(cv2.imdecode(grp[str(fid)].value, cv2.CV_LOAD_IMAGE_COLOR))
    return np.transpose(np.stack(res), (0, 3, 1, 2))


@deprecated
def StringArrayToVideo(string_array):
    """Converts a N length JPEG encoded string array to NCHW video.
    TODO: Allow partial decode (only some frames)
    """
    nframes = string_array.shape[0]
    frames = [StringArrayElementToFrame(string_array, i) for i in range(nframes)]
    video = np.transpose(np.stack(frames), (0, 3, 1, 2))
    return video


@deprecated
def StringArrayElementToFrame(string_array, i):
    return cv2.imdecode(np.fromstring(
        string_array[i].tostring(), np.uint8), cv2.CV_LOAD_IMAGE_COLOR)
