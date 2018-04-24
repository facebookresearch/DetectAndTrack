##############################################################
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from core.config import cfg

import modeling.ResNet as ResNet


# ---------------------------------------------------------------------------- #
# Keypoint heads
# ---------------------------------------------------------------------------- #

def add_ResNet_roi_conv5_head_for_keypoints(
        model, blob_in, dim_in, spatial_scale):
    model.RoIFeatureTransform(
        blob_in, '_[pose]_pool5',
        blob_rois='keypoint_rois',
        method=cfg.KRCNN.ROI_XFORM_METHOD,
        resolution=cfg.KRCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.KRCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)
    s, dim_in = ResNet.add_stage(
        model, '_[pose]_res5', '_[pose]_pool5',
        3, dim_in, 2048, 512, cfg.KRCNN.DILATION,
        stride_init=int(cfg.KRCNN.ROI_XFORM_RESOLUTION / 7))
    return s, 2048, spatial_scale


def add_roi_pose_head_v1convX(model, blob_in, dim_in, spatial_scale, nd=False):
    hidden_dim = cfg.KRCNN.CONV_HEAD_DIM
    kernel_size = cfg.KRCNN.CONV_HEAD_KERNEL
    pad_size = kernel_size // 2
    current = model.RoIFeatureTransform(
        blob_in, '_[pose]_roi_feat',
        blob_rois='keypoint_rois',
        method=cfg.KRCNN.ROI_XFORM_METHOD,
        resolution=cfg.KRCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.KRCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    for i in range(cfg.KRCNN.NUM_STACKED_CONVS):
        if nd:
            current = model.ConvNd(
                current, 'conv_fcn' + str(i + 1), dim_in, hidden_dim,
                [cfg.VIDEO.TIME_KERNEL_DIM.HEAD_KPS, kernel_size, kernel_size],
                pads=2 * [cfg.VIDEO.TIME_KERNEL_DIM.HEAD_KPS // 2, pad_size, pad_size],
                strides=[1, 1, 1],
                weight_init=(cfg.KRCNN.CONV_INIT, {'std': 0.01}),
                bias_init=('ConstantFill', {'value': 0.}))
        else:
            current = model.Conv(
                current, 'conv_fcn' + str(i + 1), dim_in, hidden_dim,
                kernel_size, stride=1, pad=pad_size,
                weight_init=(cfg.KRCNN.CONV_INIT, {'std': 0.01}),
                bias_init=('ConstantFill', {'value': 0.}))
        current = model.Relu(current, current)
        dim_in = hidden_dim

    return current, hidden_dim, spatial_scale


def add_roi_pose_head_v1convX_3d(model, blob_in, dim_in, spatial_scale):
    return add_roi_pose_head_v1convX(model, blob_in, dim_in, spatial_scale, nd=True)
