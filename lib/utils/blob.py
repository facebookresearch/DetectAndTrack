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

"""Blob helper functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import cPickle as pickle
from caffe2.python import scope
from caffe2.proto import caffe2_pb2
import numpy as np
import cv2

from core.config import cfg
import utils.image as image_utils

# OpenCL is enabled by default in OpenCV3 and it is not thread-safe leading
# to huge GPU memory allocations. See https://fburl.com/9d7tvusd
try:
  cv2.ocl.setUseOpenCL(False)
except AttributeError:
  pass


def im_list_to_blob(ims):
    """Convert a list of images into a network input.

    Assumes images are already prepared (means subtracted, BGR order, ...).
    """
    max_shape = np.array([im.shape for im in ims]).max(axis=0)
    # Pad the image so they can be divisible by a stride
    if cfg.FPN.FPN_ON:
        stride = float(cfg.FPN.COARSEST_STRIDE)
        max_shape[0] = int(np.ceil(max_shape[0] / stride) * stride)
        max_shape[1] = int(np.ceil(max_shape[1] / stride) * stride)

    num_images = len(ims)
    blob = np.zeros((num_images, max_shape[0], max_shape[1], 3),
                    dtype=np.float32)
    for i in range(num_images):
        im = ims[i]
        blob[i, 0:im.shape[0], 0:im.shape[1], :] = im
    # Move channels (axis 3) to axis 1
    # Axis order will become: (batch elem, channel, height, width)
    channel_swap = (0, 3, 1, 2)
    blob = blob.transpose(channel_swap)

    if cfg.MODEL.VIDEO_ON:
        # Now this would contain a large number of images, each of which is
        # a frame. I need to concat the images on to the time dimension
        blob = image_utils.move_batch_to_time(blob, cfg.VIDEO.NUM_FRAMES)

    return blob


def prep_im_for_blob(im, pixel_means, target_sizes, max_size):
    """Mean subtract and scale an image for use in a blob."""
    im = im.astype(np.float32, copy=False)
    im -= pixel_means
    im_shape = im.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    ims = []
    im_scales = []
    for target_size in target_sizes:
        im_scale = float(target_size) / float(im_size_min)
        # Prevent the biggest axis from being more than MAX_SIZE
        if np.round(im_scale * im_size_max) > max_size:
            im_scale = float(max_size) / float(im_size_max)
        im = cv2.resize(im, None, None, fx=im_scale, fy=im_scale,
                        interpolation=cv2.INTER_LINEAR)
        ims.append(im)
        im_scales.append(im_scale)
    return ims, im_scales


def zeros(shape, int32=False):
    """Returns a blob of the given shape with the correct data type."""
    return np.zeros(shape, dtype=np.int32 if int32 else np.float32)


def ones(shape, int32=False):
    """Returns a blob of the given shape with the correct data type."""
    return np.ones(shape, dtype=np.int32 if int32 else np.float32)


def py_op_copy_blob(blob_in, blob_out):
    needs_int32_init = False
    try:
        _ = blob.data.dtype  # noqa
    except Exception:
        needs_int32_init = blob_in.dtype == np.int32
    if needs_int32_init:
        # init can only take a list (failed on tuple)
        blob_out.init(list(blob_in.shape), caffe2_pb2.TensorProto.INT32)
    else:
        blob_out.reshape(blob_in.shape)
    blob_out.data[...] = blob_in


def unscope_name(possibly_scoped_name):
    return possibly_scoped_name[
        possibly_scoped_name.rfind(scope._NAMESCOPE_SEPARATOR) + 1:]


def get_loss_gradients(model, loss_blobs):
    """Generate a gradient of 1 for each loss specified in 'loss_blobs'"""
    loss_gradients = {}
    for b in loss_blobs:
        loss_grad = model.net.ConstantFill(b, [b + '_grad'], value=1.0)
        loss_gradients[str(b)] = str(loss_grad)
    return loss_gradients


def serialize(obj):
    """Serialize a Python object using pickle and encode it as an array of
    float32 values so that it can be feed into the workspace. See deserialize().
    """
    return np.fromstring(pickle.dumps(obj), dtype=np.uint8).astype(np.float32)


def deserialize(arr):
    """Unserialize a Python object from an array of float32 values fetched from
    a workspace. See serialize().
    """
    return pickle.loads(arr.astype(np.uint8).tobytes())
