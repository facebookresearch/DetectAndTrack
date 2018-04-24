##############################################################
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################

"""Image helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cv2
import numpy as np
import os.path as osp

import utils.video_io as video_io_utils


def aspect_ratio_rel(im, aspect_ratio):
    """Performs width-relative aspect ratio transformation."""
    im_h, im_w = im.shape[:2]
    im_ar_w = int(round(aspect_ratio * im_w))
    im_ar = cv2.resize(im, dsize=(im_ar_w, im_h))
    return im_ar


def aspect_ratio_abs(im, aspect_ratio):
    """Performs absolute aspect ratio transformation."""
    im_h, im_w = im.shape[:2]
    im_area = im_h * im_w

    im_ar_w = np.sqrt(im_area * aspect_ratio)
    im_ar_h = np.sqrt(im_area / aspect_ratio)
    assert np.isclose(im_ar_w / im_ar_h, aspect_ratio)

    im_ar = cv2.resize(im, dsize=(int(im_ar_w), int(im_ar_h)))
    return im_ar


def get_image_path(entry):
    im_names = entry['image']
    if isinstance(im_names, list):
        return im_names[len(im_names) // 2]
    return im_names


def _read_image(impaths, frame_ids, entry):
    assert len(impaths) > 0, 'No images passed for reading'
    _, filext = osp.splitext(impaths[0])
    if not entry['dataset'].frames_from_video:
        img = [cv2.imread(impath) for impath in impaths]
    else:
        # Need to subtract 1 as frame_ids are 1-indexed in the ROIDB, and the
        # video reader requires 0-indexed.
        img = [im for im in video_io_utils.ReadClipFromVideo(
            impaths[0], [el - 1 for el in frame_ids],
            width=entry['width'], height=entry['height'])]
    return img


def read_image_video(entry, key_frame_only=False):
    """Given a roidb entry, read the image or video as the case demands.
    Return a list of images. For the single image case, this would be a 1
    element list."""
    if isinstance(entry['image'], list):
        impaths = entry['image']
        frame_ids = entry['all_frame_ids']
        if key_frame_only:
            impaths = [get_image_path(entry)]
            frame_ids = [frame_ids[len(frame_ids) // 2]]
        ims = _read_image(impaths, frame_ids, entry)
    else:
        # Single image (1 frame video)
        ims = _read_image([entry['image']], [entry['frame_id']], entry)
    return ims


def move_batch_to_time(blob, num_frames):
    """ Given a B*TxCxHxW blob, convert to BxCxTxHxW. """
    T = num_frames
    assert(blob.shape[0] % T == 0)
    # Reshape from B*TxCxHxW to 1xB*TxCxHxW
    blob = np.expand_dims(blob, axis=0)
    # Reshape from 1xB*TxCxHxW to BxTxCxHxW
    blob = np.reshape(
        blob, (-1, T, blob.shape[-3], blob.shape[-2], blob.shape[-1]))
    # Swap time and channel dimension to get BxCxTxHxW
    blob = np.transpose(blob, (0, 2, 1, 3, 4))
    return blob
