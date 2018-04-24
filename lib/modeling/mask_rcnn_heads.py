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
# Mask heads
# ---------------------------------------------------------------------------- #

# v1up: conv 3x3, conv 3x3, deconv 2x2
# abused name (not related to ResNet)
def ResNet_mask_rcnn_fcn_head_v1up4convs(
        model, blob_in, dim_in, spatial_scale, preprefix='_[mask]_'):
    roi_feat = model.RoIFeatureTransform(
        blob_in, blob_out=preprefix + 'roi_feat',
        blob_rois='mask_rois',
        method=cfg.MRCNN.ROI_XFORM_METHOD,
        resolution=cfg.MRCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.MRCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    dim_inner = cfg.MRCNN.DIM_REDUCED

    # two 3x3 layers:
    dilation = cfg.MRCNN.DILATION

    model.Conv(
        roi_feat, preprefix + 'fcn1', dim_in, dim_inner, 3,
        pad=1 * dilation, stride=1, dilation=dilation,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),  # std only for gauss
        bias_init=('ConstantFill', {'value': 0.}))
    model.Relu(preprefix + 'fcn1', preprefix + 'fcn1')

    model.Conv(
        preprefix + 'fcn1', preprefix + 'fcn2', dim_inner, dim_inner, 3,
        pad=1 * dilation, stride=1, dilation=dilation,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),  # std only for gauss
        bias_init=('ConstantFill', {'value': 0.}))
    model.Relu(preprefix + 'fcn2', preprefix + 'fcn2')

    model.Conv(
        preprefix + 'fcn2', preprefix + 'fcn3', dim_inner, dim_inner, 3,
        pad=1 * dilation, stride=1, dilation=dilation,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),  # std only for gauss
        bias_init=('ConstantFill', {'value': 0.}))
    model.Relu(preprefix + 'fcn3', preprefix + 'fcn3')

    model.Conv(
        preprefix + 'fcn3', preprefix + 'fcn4', dim_inner, dim_inner, 3,
        pad=1 * dilation, stride=1, dilation=dilation,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),  # std only for gauss
        bias_init=('ConstantFill', {'value': 0.}))
    model.Relu(preprefix + 'fcn4', preprefix + 'fcn4')

    # upsample layer
    model.ConvTranspose(
        preprefix + 'fcn4', 'conv5_mask', dim_inner, dim_inner, 2,
        pad=0, stride=2,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),  # std only for gauss
        bias_init=('ConstantFill', {'value': 0.}))
    blob_mask = model.Relu('conv5_mask', 'conv5_mask')

    return blob_mask, dim_inner, spatial_scale


# v1up: conv 3x3, conv 3x3, deconv 2x2
# abused name (not related to ResNet)
def ResNet_mask_rcnn_fcn_head_v1up(
        model, blob_in, dim_in, spatial_scale, preprefix='_[mask]_'):
    roi_feat = model.RoIFeatureTransform(
        blob_in, blob_out=preprefix + 'roi_feat',
        blob_rois='mask_rois',
        method=cfg.MRCNN.ROI_XFORM_METHOD,
        resolution=cfg.MRCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.MRCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    dim_inner = cfg.MRCNN.DIM_REDUCED

    # two 3x3 layers:
    dilation = cfg.MRCNN.DILATION

    model.Conv(
        roi_feat, preprefix + 'fcn1', dim_in, dim_inner, 3,
        pad=1 * dilation, stride=1, dilation=dilation,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),  # std only for gauss
        bias_init=('ConstantFill', {'value': 0.}))
    model.Relu(preprefix + 'fcn1', preprefix + 'fcn1')

    model.Conv(
        preprefix + 'fcn1', preprefix + 'fcn2', dim_inner, dim_inner, 3,
        pad=1 * dilation, stride=1, dilation=dilation,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),  # std only for gauss
        bias_init=('ConstantFill', {'value': 0.}))
    model.Relu(preprefix + 'fcn2', preprefix + 'fcn2')

    # upsample layer
    model.ConvTranspose(
        preprefix + 'fcn2', 'conv5_mask', dim_inner, dim_inner, 2,
        pad=0, stride=2,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),  # std only for gauss
        bias_init=('ConstantFill', {'value': 0.}))
    blob_mask = model.Relu('conv5_mask', 'conv5_mask')

    return blob_mask, dim_inner, spatial_scale


# v0upshare: conv5, deconv 2x2, sharing weights and computation
def ResNet_mask_rcnn_fcn_head_v0upshare(
        model, blob_in, dim_in, spatial_scale, preprefix='_[mask]_'):
    """At traininig time, only bbox branch rois (RESOLUTION_ROI_BBOX) are used;
    at testing time, the mask branch rois (RESOLUTION_ROI) are used.
    To have consistent roi features, assert the roi settings are the same.
    """
    assert cfg.MRCNN.ROI_XFORM_RESOLUTION == cfg.FAST_RCNN.ROI_XFORM_RESOLUTION

    if model.train:  # share computation with bbox head at training time
        dim_conv5 = 2048
        blob_conv5 = model.net.SampleAs(
            ['res5_2_sum', 'roi_has_mask_int32'],
            [preprefix + 'res5_2_sum_sliced'])
    else:  # re-compute at test time
        blob_conv5, dim_conv5, spatial_scale = \
            add_ResNet_roi_conv5_head_for_masks(
                model, blob_in, dim_in, spatial_scale, preprefix,
                dilation=cfg.MRCNN.DILATION, shared=False)

    dim_reduced = cfg.MRCNN.DIM_REDUCED

    blob_mask = model.ConvTranspose(
        blob_conv5, 'conv5_mask', dim_conv5, dim_reduced, 2,
        pad=0, stride=2,
        weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),  # std only for gauss
        bias_init=('ConstantFill', {'value': 0.}))
    model.Relu('conv5_mask', 'conv5_mask')

    return blob_mask, dim_reduced, spatial_scale


# v0up: conv5, deconv 2x2, no sharing
def ResNet_mask_rcnn_fcn_head_v0up(
        model, blob_in, dim_in, spatial_scale, preprefix='_[mask]_'):

    blob_conv5, dim_conv5, spatial_scale_conv5 = \
        add_ResNet_roi_conv5_head_for_masks(
            model, blob_in, dim_in, spatial_scale, preprefix,
            dilation=cfg.MRCNN.DILATION, shared=False)

    dim_reduced = cfg.MRCNN.DIM_REDUCED

    model.ConvTranspose(
        blob_conv5, 'conv5_mask', dim_conv5, dim_reduced, 2,
        pad=0, stride=2,
        weight_init=('GaussianFill', {'std': 0.001}),
        bias_init=('ConstantFill', {'value': 0.}))
    blob_mask = model.Relu('conv5_mask', 'conv5_mask')

    return blob_mask, dim_reduced, spatial_scale_conv5


def add_ResNet_roi_conv5_head_for_masks(
        model, blob_in, dim_in, spatial_scale, preprefix='_[mask]_',
        dilation=1, shared=False):
    assert not shared, \
        'Using shared ResNet stage not supported (temporarily)'
    model.RoIFeatureTransform(
        blob_in,
        blob_out=preprefix + 'pool5',
        blob_rois='mask_rois',
        method=cfg.MRCNN.ROI_XFORM_METHOD,
        resolution=cfg.MRCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.MRCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    stride_init = int(cfg.MRCNN.ROI_XFORM_RESOLUTION / 7)  # by default: 2
    if not shared:
        s, dim_in = ResNet.add_stage(
            model, preprefix + 'res5', preprefix + 'pool5',
            3, dim_in, 2048, 512, dilation, stride_init=stride_init)
    else:
        s, dim_in = ResNet.add_stage_shared(
            model, preprefix, 'res5', preprefix + 'pool5',
            3, dim_in, 2048, 512, dilation, stride_init=stride_init)

    return s, 2048, spatial_scale
