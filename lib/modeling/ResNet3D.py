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

from modeling.common import ConvStageInfo
from core.config import cfg


# ------------------------------------------------------------------------------
# various transformations (may expand and may consider a new helper)
# ------------------------------------------------------------------------------
def bottleneck_transformation(
        model, blob_in, dim_in, dim_out,
        stride, prefix, dim_inner, dilation=1, group=1,
        time_kernel_dim=1, time_stride_on=False):
    """
    In original resnet, stride=2 is on 1x1.
    In fb.torch resnet, stride=2 is on 3x3.
    """
    (str1x1, str3x3) = (stride, 1) if cfg.RESNETS.STRIDE_1X1 else (1, stride)

    # conv 1x1 -> BN -> ReLU
    cur = model.ConvAffineNd(
        blob_in, prefix + '_branch2a', dim_in, dim_inner, kernels=[1, 1, 1],
        strides=[
            str1x1 if time_stride_on else 1, str1x1, str1x1],
        pads=2 * [0, 0, 0], inplace=True)
    cur = model.Relu(cur, cur)

    # conv 3x3 -> BN -> ReLU
    cur = model.ConvAffineNd(
        cur, prefix + '_branch2b', dim_inner, dim_inner,
        kernels=[time_kernel_dim, 3, 3],
        strides=[
            str3x3 if time_stride_on else 1, str3x3, str3x3],
        pads=2 * [time_kernel_dim // 2, dilation, dilation],
        dilations=[1, dilation, dilation], group=group, inplace=True)
    cur = model.Relu(cur, cur)

    # conv 1x1 -> BN (no ReLU)
    # NB: for now this AffineChannel op cannot be in-place due to a bug in C2
    # gradient computation for graphs like this
    cur = model.ConvAffineNd(
        cur, prefix + '_branch2c', dim_inner, dim_out, kernels=[1, 1, 1],
        strides=[1, 1, 1], pads=2 * [0, 0, 0], inplace=False)
    return cur


# 3x3-3x3 branch (vgg-like), used for R-18,34
def basic_transformation(
        model, blob_in, dim_in, dim_out, stride, prefix, dim_inner,
        dilation=1, group=1, time_kernel_dim=1, time_stride_on=False):
    """
    basic block is mostly deprecated and untested
    """
    if dim_inner is None:
        dim_inner = dim_out

    # 3x3 layer
    blob_out = model.ConvAffineNd(
        blob_in, prefix + "_branch2a", dim_in, dim_inner,
        kernels=[time_kernel_dim, 3, 3],
        strides=[stride if time_stride_on else 1, stride, stride],
        pads=2 * [time_kernel_dim // 2, 1, 1], inplace=True)
    blob_out = model.Relu(blob_out, blob_out)

    # 3x3 layer (no relu)
    blob_out = model.ConvAffineNd(
        blob_out, prefix + "_branch2b", dim_inner, dim_out,
        kernels=[time_kernel_dim, 3, 3],
        strides=[1, 1, 1], pads=2 * [time_kernel_dim // 2, dilation, dilation],
        dilations=dilation, group=group, inplace=False)
    return blob_out


# ---------------------------------------------------------------------------- #
# Generic ResNet components
# ---------------------------------------------------------------------------- #

def add_shortcut(model, prefix, blob_in, dim_in, dim_out, stride,
                 time_stride_on):
    if dim_in == dim_out:
        return blob_in
    c = model.ConvNd(
        blob_in, prefix + '_branch1',
        dim_in, dim_out,
        [1, 1, 1],
        strides=[
            stride if time_stride_on else 1, stride, stride],
        pads=2 * [0, 0, 0],
        no_bias=1)
    return model.AffineChannelNd(c, prefix + '_branch1_bn', dim_out=dim_out)


# TODO(km): revisit this
# def add_shortcut_shared(
#         model, layer_prefix, weight_prefix, blob_in, dim_in, dim_out, stride):
#     if dim_in == dim_out:
#         return blob_in
#     c = model.ConvShared(
#         blob_in,
#         layer_prefix + '_branch1',
#         dim_in, dim_out, kernel=1, stride=stride,
#         weight=weight_prefix + '_branch1_w',
#         no_bias=1)
#     return model.AffineChannel(
#         c, layer_prefix + '_branch1_bn',
#         share_with=weight_prefix + '_branch1_bn')


def add_bottleneck_block(
        stage_id,
        model, prefix, blob_in, dim_in, dim_out, dim_inner, dilation,
        stride_init=2, inplace_sum=False,
        time_kernel_dim=1, time_stride_on=False):
    """Adds a single bottleneck block."""

    # prefix = res<stage>_<sub_stage>, e.g., res2_3

    # Max pooling is performed prior to the first stage (which is uniquely
    # distinguished by dim_in = 64), thus we keep stride = 1 for the first stage
    # stride = stride_init if (
    #     dim_in != dim_out and dim_in != 64 and dilation == 1) else 1
    # rgirdhar: The above didn't work for R-18/34 case where the first two
    # stages have 64-D outputs. So, using an explicit stage_id instead.
    stride = stride_init if (
        dim_in != dim_out and stage_id != 1 and dilation == 1) else 1

    # transformation blob
    tr = globals()[cfg.RESNETS.TRANS_FUNC](
        model, blob_in, dim_in, dim_out, stride, prefix, dim_inner,
        group=cfg.RESNETS.NUM_GROUPS, dilation=dilation,
        time_kernel_dim=time_kernel_dim, time_stride_on=time_stride_on)

    # sum -> ReLU
    sc = add_shortcut(model, prefix, blob_in, dim_in, dim_out, stride,
                      time_stride_on=time_stride_on)
    if inplace_sum:
        s = model.net.Sum([tr, sc], tr)
    else:
        s = model.net.Sum([tr, sc], prefix + '_sum')

    return model.Relu(s, s)


# TODO(km): revisit this
# def add_bottleneck_block_shared(
#         model, layer_prefix, weight_prefix, blob_in, dim_in, dim_out, dim_inner,
#         dilation, stride_init=2, inplace_sum=False):
#     # Max pooling is performed prior to the first bottleneck block_cfg
#     # (when dim_in = 64)
#     stride = stride_init if (
#         dim_in != dim_out and dim_in != 64 and dilation == 1) else 1
#     # prefix = res<stage><sub_stage>, e.g., res2a
#     # conv 1x1 -> BN -> ReLU
#     c = model.ConvShared(
#         blob_in,
#         layer_prefix + '_branch2a',
#         dim_in, dim_inner, kernel=1, stride=stride,
#         weight=weight_prefix + '_branch2a_w',
#         no_bias=1)
#     bn = model.AffineChannel(
#         c,
#         layer_prefix + '_branch2a_bn',
#         share_with=weight_prefix + '_branch2a_bn')
#     r = model.Relu(bn, bn)
#     # conv 3x3 -> BN -> ReLU
#     c = model.ConvShared(
#         r,
#         layer_prefix + '_branch2b',
#         dim_inner, dim_inner, kernel=3, stride=1, pad=1 * dilation,
#         dilation=dilation,
#         weight=weight_prefix + '_branch2b_w',
#         no_bias=1)
#     bn = model.AffineChannel(
#         c,
#         layer_prefix + '_branch2b_bn',
#         share_with=weight_prefix + '_branch2b_bn')
#     r = model.Relu(bn, bn)
#     # conv 1x1 -> BN (no ReLU)
#     c = model.ConvShared(
#         r,
#         layer_prefix + '_branch2c',
#         dim_inner, dim_out, kernel=1, stride=1,
#         weight=weight_prefix + '_branch2c_w',
#         no_bias=1)
#     bn = model.AffineChannel(
#         c, layer_prefix + '_branch2c_bn',
#         share_with=weight_prefix + '_branch2c_bn')
#     # sum -> ReLU
#     sc = add_shortcut_shared(
#         model, layer_prefix, weight_prefix,
#         blob_in, dim_in, dim_out, stride)
#     if inplace_sum:
#         s = model.net.Sum([bn, sc], bn)
#     else:
#         s = model.net.Sum([bn, sc], layer_prefix + '_sum')
#     return model.Relu(s, s)


def add_stage(
        stage_id,
        model, prefix, blob_in, n, dim_in, dim_out, dim_inner, dilation,
        stride_init=2, time_kernel_dim=1, time_stride_on=False):
    """Adds a single ResNet stage by stacking n bottleneck blocks."""
    # e.g., prefix = res2
    for i in range(n):
        blob_in = add_bottleneck_block(
            stage_id,
            model, '{}_{}'.format(prefix, i),
            blob_in, dim_in, dim_out, dim_inner,
            dilation, stride_init,
            # Not using inplace for the last block;
            # it may be fetched externally or used by FPN
            inplace_sum=i < n - 1,
            time_kernel_dim=time_kernel_dim,
            time_stride_on=time_stride_on)
        dim_in = dim_out
    return blob_in, dim_in


# TODO(km): revisit this
# def add_stage_shared(
#         model, preprefix, prefix, blob_in, n, dim_in, dim_out, dim_inner,
#         dilation, stride_init=2):
#     # e.g., preprefix = _[mask]_
#     # e.g., prefix = res2
#     for i in range(n):
#         blob_in = add_bottleneck_block_shared(
#             model,
#             '{}_{}'.format(preprefix + prefix, i),  # e.g., _[mask]_res5
#             '{}_{}'.format(prefix, i),  # e.g., res5
#             blob_in, dim_in, dim_out, dim_inner,
#             dilation, stride_init,
#             # Not using inplace for the last block;
#             # it may be fetched externally or used by FPN
#             inplace_sum=i < n - 1)
#         dim_in = dim_out
#     return blob_in, dim_in


def add_ResNet_convX_body(model, block_counts, freeze_at=2,
                          feat_dims=(64, 256, 512, 1024, 2048)):
    """Adds a ResNet body from input data up through the res5 (aka conv5) stage.
    The final res5/conv5 stage may be optionally excluded (hence convX, where
    X = 4 or 5)."""
    assert freeze_at in [0, 2, 3, 4, 5]
    # No striding or temporal convolutions at layer 1
    p = model.ConvNd('data', 'conv1', 3, feat_dims[0], [1, 7, 7],
                     pads=2 * [0, 3, 3],
                     strides=[1, 2, 2],
                     no_bias=1)
    p = model.AffineChannelNd(p, 'res_conv1_bn', dim_out=feat_dims[0], inplace=True)
    p = model.Relu(p, p)
    p = model.MaxPool(p, 'pool1', kernels=[1, 3, 3],
                      pads=2 * [0, 1, 1], strides=[1, 2, 2])
    dim_in = feat_dims[0]
    dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
    (n1, n2, n3) = block_counts[:3]
    # No striding or temporal conv at layer 2 either
    s, dim_in = add_stage(
        1, model, 'res2', p, n1, dim_in, feat_dims[1], dim_bottleneck, 1,
        time_kernel_dim=1, time_stride_on=False)
    if freeze_at == 2:
        model.StopGradient(s, s)
    s, dim_in = add_stage(
        2, model, 'res3', s, n2, dim_in, feat_dims[2], dim_bottleneck * 2, 1,
        time_kernel_dim=cfg.VIDEO.TIME_KERNEL_DIM.BODY,
        time_stride_on=cfg.VIDEO.TIME_STRIDE_ON)
    if freeze_at == 3:
        model.StopGradient(s, s)
    s, dim_in = add_stage(
        3, model, 'res4', s, n3, dim_in, feat_dims[3], dim_bottleneck * 4, 1,
        time_kernel_dim=cfg.VIDEO.TIME_KERNEL_DIM.BODY,
        time_stride_on=cfg.VIDEO.TIME_STRIDE_ON)
    if freeze_at == 4:
        model.StopGradient(s, s)
    if len(block_counts) == 4:
        n4 = block_counts[3]
        s, dim_in = add_stage(
            4, model, 'res5', s, n4, dim_in, feat_dims[4], dim_bottleneck * 8,
            cfg.MODEL.DILATION,
            time_kernel_dim=cfg.VIDEO.TIME_KERNEL_DIM.BODY,
            time_stride_on=cfg.VIDEO.TIME_STRIDE_ON)
        if freeze_at == 5:
            model.StopGradient(s, s)
        return s, dim_in, 1. / 32. * cfg.MODEL.DILATION
    else:
        return s, dim_in, 1. / 16.


def add_ResNet_roi_conv5_head(model, blob_in, dim_in, spatial_scale,
                              block_counts=3, dim_out=2048):
    """Adds an RoI feature transformation (e.g., RoI pooling) followed by a
    res5/conv5 head applied to each RoI."""
    # TODO(rbg): This contains Fast R-CNN specific config options making it non-
    # reusable; make this more generic with model-specific wrappers
    model.RoIFeatureTransform(
        blob_in, 'pool5',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=cfg.FAST_RCNN.ROI_XFORM_RESOLUTION,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)
    dim_bottleneck = cfg.RESNETS.NUM_GROUPS * cfg.RESNETS.WIDTH_PER_GROUP
    stride_init = int(cfg.FAST_RCNN.ROI_XFORM_RESOLUTION / 7)
    s, dim_in = add_stage(
        4, model, 'res5', 'pool5', block_counts, dim_in, dim_out,
        dim_bottleneck * 8, 1, stride_init)
    # s = model.AveragePool(s, 'res5_pool', kernel=7)
    # Reduce mean across all dimensions (h,w,t)
    model.ReduceBackMean(s, 'res5_pool_w')
    model.ReduceBackMean('res5_pool_w', 'res5_pool')
    # Do not pool on time as well, as I use a 3D head on top, so leave it as a
    # 3D blob so I can do 3D conv on it.
    # s = model.ReduceBackMean('res5_pool', 'res5_pool')
    s = model.ExpandDims('res5_pool', 'res5_pool', dims=[3, 4])
    return s, dim_out, spatial_scale


# ---------------------------------------------------------------------------- #
# Bits for specific architectures (ResNet50, ResNet101, ...)
# ---------------------------------------------------------------------------- #

def add_ResNet18_conv4_body(model):
    cfg.RESNETS.TRANS_FUNC = 'basic_transformation'
    return add_ResNet_convX_body(model, (2, 2, 2), freeze_at=2,
                                 feat_dims=(64, 64, 128, 256))


def add_ResNet18_roi_conv5_head(*args, **kwargs):
    """ Usable with R18/34 models. """
    kwargs['dim_out'] = 512
    kwargs['block_counts'] = 2
    return add_ResNet_roi_conv5_head(*args, **kwargs)


def add_ResNet18_conv5_body(model):
    cfg.RESNETS.TRANS_FUNC = 'basic_transformation'
    return add_ResNet_convX_body(model, (2, 2, 2, 2), freeze_at=2,
                                 feat_dims=(64, 64, 128, 256, 512))


def add_ResNet34_conv4_body(model):
    cfg.RESNETS.TRANS_FUNC = 'basic_transformation'
    return add_ResNet_convX_body(model, (3, 4, 6), freeze_at=2,
                                 feat_dims=(64, 64, 128, 256))


def add_ResNet34_roi_conv5_head(*args, **kwargs):
    """ Usable with R18/34 models. """
    kwargs['dim_out'] = 512
    kwargs['block_counts'] = 3
    return add_ResNet_roi_conv5_head(*args, **kwargs)


def add_ResNet34_conv5_body(model):
    cfg.RESNETS.TRANS_FUNC = 'basic_transformation'
    return add_ResNet_convX_body(model, (3, 4, 6, 3), freeze_at=2,
                                 feat_dims=(64, 64, 128, 256, 512))


def add_ResNet50_conv4_body(model):
    cfg.RESNETS.TRANS_FUNC = 'bottleneck_transformation'
    return add_ResNet_convX_body(model, (3, 4, 6), freeze_at=2)


def add_ResNet50_conv5_body(model):
    cfg.RESNETS.TRANS_FUNC = 'bottleneck_transformation'
    return add_ResNet_convX_body(model, (3, 4, 6, 3), freeze_at=2)


def add_ResNet101_conv4_body(model):
    cfg.RESNETS.TRANS_FUNC = 'bottleneck_transformation'
    return add_ResNet_convX_body(model, (3, 4, 23), freeze_at=2)


def add_ResNet101_conv5_body(model):
    cfg.RESNETS.TRANS_FUNC = 'bottleneck_transformation'
    return add_ResNet_convX_body(model, (3, 4, 23, 3), freeze_at=2)


def add_ResNet152_conv5_body(model):
    cfg.RESNETS.TRANS_FUNC = 'bottleneck_transformation'
    return add_ResNet_convX_body(model, (3, 8, 36, 3), freeze_at=2)


# ---------------------------------------------------------------------------- #
# Stage info
# ---------------------------------------------------------------------------- #

def stage_info_ResNet18_conv5():
    return ConvStageInfo(
        blobs=('res5_1_sum', 'res4_1_sum', 'res3_1_sum', 'res2_1_sum'),
        dims=(512, 256, 128, 64),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.))


def stage_info_ResNet34_conv5():
    return ConvStageInfo(
        blobs=('res5_2_sum', 'res4_5_sum', 'res3_3_sum', 'res2_2_sum'),
        dims=(512, 256, 128, 64),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.))


def stage_info_ResNet50_conv5():
    return ConvStageInfo(
        blobs=('res5_2_sum', 'res4_5_sum', 'res3_3_sum', 'res2_2_sum'),
        dims=(2048, 1024, 512, 256),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.))


def stage_info_ResNet101_conv5():
    return ConvStageInfo(
        blobs=('res5_2_sum', 'res4_22_sum', 'res3_3_sum', 'res2_2_sum'),
        dims=(2048, 1024, 512, 256),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.))


def stage_info_ResNet152_conv5():
    return ConvStageInfo(
        blobs=('res5_2_sum', 'res4_35_sum', 'res3_7_sum', 'res2_2_sum'),
        dims=(2048, 1024, 512, 256),
        spatial_scales=(1. / 32., 1. / 16., 1. / 8., 1. / 4.))
