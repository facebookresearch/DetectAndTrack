##############################################################
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from core.config import cfg
import utils.blob as blob_utils
import roi_data.rpn
import roi_data.fast_rcnn
import utils.image as image_utils

import logging
logger = logging.getLogger(__name__)


def get_minibatch_blob_names():
    """Returns blob names in the order in which they are read by the data
    loader.
    """
    # data blob: holds a batch of N images, each with 3 channels
    blob_names = ['data']
    if cfg.RPN.ON:
        # RPN-only or end-to-end Faster R-CNN
        blob_names += roi_data.rpn.get_rpn_blob_names()
    else:
        # Fast R-CNN like models trained on precomputed proposals
        blob_names += roi_data.fast_rcnn.get_fast_rcnn_blob_names()
    return blob_names


def get_minibatch(roidb):
    """Given a roidb, construct a minibatch sampled from it."""
    # We collect blobs from each image onto a list and then concat them into a
    # single tensor, hence we initialize each blob to an empty list
    blobs = {k: [] for k in get_minibatch_blob_names()}
    # Get the input image blob, formatted for caffe2
    im_blob, im_scales = _get_image_blob(roidb)
    blobs['data'] = im_blob
    if cfg.RPN.ON:
        # RPN-only or end-to-end Faster R-CNN
        valid = roi_data.rpn.add_rpn_blobs(blobs, im_scales, roidb)
    else:
        # Fast R-CNN like models trained on precomputed proposals
        valid = roi_data.fast_rcnn.add_fast_rcnn_blobs(blobs, im_scales, roidb)
    return blobs, valid


def _get_image_blob(roidb):
    """Builds an input blob from the images in the roidb at the specified
    scales.
    """
    num_images = len(roidb)
    # Sample random scales to use for each image in this batch
    scale_inds = np.random.randint(
        0, high=len(cfg.TRAIN.SCALES), size=num_images)
    processed_ims = []
    im_scales = []
    for i in range(num_images):
        ims = image_utils.read_image_video(roidb[i])
        for im_id, im in enumerate(ims):
            if roidb[i]['flipped']:
                im = im[:, ::-1, :]
            target_size = cfg.TRAIN.SCALES[scale_inds[i]]
            im, im_scale = blob_utils.prep_im_for_blob(
                im, cfg.PIXEL_MEANS, [target_size], cfg.TRAIN.MAX_SIZE)
            ims[im_id] = im[0]
        # Just taking the im_scale for the last im in ims is fine (all are same)
        im_scales.append(im_scale[0])
        processed_ims += ims

    # Create a blob to hold the input images
    blob = blob_utils.im_list_to_blob(processed_ims)
    return blob, im_scales
