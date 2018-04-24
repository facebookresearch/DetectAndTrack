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
from modeling.generate_anchors import generate_anchors
import modeling.ResNet3D as ResNet

# Lowest and highest levels in the default feature pyramid network
LOWEST_LVL = 2
HIGHEST_LVL = 5


# ---------------------------------------------------------------------------- #
# RPN with ResNet
# ---------------------------------------------------------------------------- #

def add_fpn_ResNet18_conv5_body(model):
    return add_fpn_generic_onto_body(
        model, ResNet.add_ResNet18_conv5_body, ResNet.stage_info_ResNet18_conv5)


def add_fpn_ResNet34_conv5_body(model):
    return add_fpn_generic_onto_body(
        model, ResNet.add_ResNet34_conv5_body, ResNet.stage_info_ResNet34_conv5)


def add_fpn_ResNet50_conv5_body(model):
    return add_fpn_generic_onto_body(
        model, ResNet.add_ResNet50_conv5_body, ResNet.stage_info_ResNet50_conv5)


def add_fpn_ResNet50_conv5_P2only_body(model):
    return add_fpn_generic_onto_body(
        model, ResNet.add_ResNet50_conv5_body, ResNet.stage_info_ResNet50_conv5,
        P2only=True)


def add_fpn_ResNet101_conv5_body(model):
    return add_fpn_generic_onto_body(
        model, ResNet.add_ResNet101_conv5_body,
        ResNet.stage_info_ResNet101_conv5)


def add_fpn_ResNet101_conv5_P2only_body(model):
    return add_fpn_generic_onto_body(
        model, ResNet.add_ResNet101_conv5_body,
        ResNet.stage_info_ResNet101_conv5, P2only=True)


def add_fpn_ResNet152_conv5_body(model):
    return add_fpn_generic_onto_body(
        model, ResNet.add_ResNet152_conv5_body,
        ResNet.stage_info_ResNet152_conv5)


def add_fpn_ResNet152_conv5_P2only_body(model):
    return add_fpn_generic_onto_body(
        model, ResNet.add_ResNet152_conv5_body,
        ResNet.stage_info_ResNet152_conv5, P2only=True)


# ---------------------------------------------------------------------------- #
# Functions for bolting FPN onto a backbone architectures
# ---------------------------------------------------------------------------- #

def get_min_max_levels():
    # Add P6, P7, ...
    min_level = LOWEST_LVL
    max_level = HIGHEST_LVL
    if cfg.FPN.MULTILEVEL_RPN and not cfg.FPN.MULTILEVEL_ROIS:
        max_level = cfg.FPN.RPN_MAX_LEVEL
        min_level = cfg.FPN.RPN_MIN_LEVEL
    if not cfg.FPN.MULTILEVEL_RPN and cfg.FPN.MULTILEVEL_ROIS:
        max_level = cfg.FPN.ROI_MAX_LEVEL
        min_level = cfg.FPN.ROI_MIN_LEVEL
    if cfg.FPN.MULTILEVEL_RPN and cfg.FPN.MULTILEVEL_ROIS:
        max_level = max(cfg.FPN.RPN_MAX_LEVEL, cfg.FPN.ROI_MAX_LEVEL)
        min_level = min(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.ROI_MIN_LEVEL)
    return min_level, max_level


def add_fpn_generic_onto_body(
        model, conv_body_func, stage_info_func, P2only=False):
    # Note: blobs_conv is in revsersed order: [fpn5, fpn4, fpn3, fpn2]
    # similarly for dims_conv: [2048, 1024, 512, 256]
    # similarly for spatial_scales_fpn: [1/32, 1/16, 1/8, 1/4]

    conv_body_func(model)
    blobs_fpn, dim_fpn, spatial_scales_fpn = add_fpn(model, stage_info_func())

    if P2only:
        # use only the finest level
        return blobs_fpn[-1], dim_fpn, spatial_scales_fpn[-1]
    else:
        # use all levels
        return blobs_fpn, dim_fpn, spatial_scales_fpn


def add_fpn(model, stage_info):
    """Adds FPN connections based on the model described in the FPN paper."""
    # Top-down: 2x up
    # Lateral: 1x1 conv with dim reduction
    # Post-hoc: 3x3
    fpn_dim = cfg.FPN.DIM
    min_level, max_level = get_min_max_levels()

    # For the coarest level: 1x1 conv only
    model.ConvNd(
        stage_info.blobs[0], 'fpn_inner_' + stage_info.blobs[0],
        stage_info.dims[0], fpn_dim, [1, 1, 1],
        pads=2 * [0, 0, 0], strides=[1, 1, 1],
        weight_init=('XavierFill', {}),
        bias_init=('ConstantFill', {'value': 0.}))

    # For other levels add top-down and lateral connections
    for i in range(len(stage_info.blobs) - 1 - (min_level - LOWEST_LVL)):
        fpn_top = 'fpn_inner_' + stage_info.blobs[i]
        fpn_lateral = stage_info.blobs[i + 1]
        fpn_bottom = 'fpn_inner_' + stage_info.blobs[i + 1]

        dim_top = fpn_dim
        dim_lateral = stage_info.dims[i + 1]

        add_topdown_lateral_module(
            model, fpn_top, fpn_lateral, fpn_bottom, dim_top, dim_lateral)

    # Post-hoc scale-specific 3x3 convs
    blobs_fpn = []
    spatial_scales = []
    for i in range(len(stage_info.blobs) - (min_level - LOWEST_LVL)):
        fpn_blob = model.ConvNd(
            'fpn_inner_' + stage_info.blobs[i], 'fpn_' + stage_info.blobs[i],
            fpn_dim, fpn_dim, [cfg.VIDEO.TIME_KERNEL_DIM.BODY, 3, 3],
            pads=2 * [cfg.VIDEO.TIME_KERNEL_DIM.BODY // 2, 1, 1],
            strides=[1, 1, 1],
            weight_init=('XavierFill', {}),
            bias_init=('ConstantFill', {'value': 0.}))
        blobs_fpn += [fpn_blob]
        spatial_scales += [stage_info.spatial_scales[i]]

    # Original FPN P6 level implementation from our CVPR'17 FPN paper
    # Check if we needs the P6 feature map
    if not cfg.FPN.EXTRA_CONV_LEVELS and max_level == HIGHEST_LVL + 1:
        P6_blob_in = blobs_fpn[0]
        P6_name = P6_blob_in + '_subsampled_2x'
        # Use max pooling to simulate stride 2 subsampling
        P6_blob = model.MaxPool(
            P6_blob_in, P6_name,
            kernels=[1, 1, 1], pads=2 * [0, 0, 0],
            strides=[1, 2, 2])
        blobs_fpn.insert(0, P6_blob)
        spatial_scales.insert(0, spatial_scales[0] * 0.5)

    # Newer style from our one-stage detection paper
    if cfg.FPN.EXTRA_CONV_LEVELS and max_level > HIGHEST_LVL:
        fpn_blob = stage_info.blobs[0]
        dim_in = stage_info.dims[0]
        for i in range(HIGHEST_LVL + 1, max_level + 1):
            if i > HIGHEST_LVL + 1:
                fpn_blob = model.Relu(fpn_blob, fpn_blob)
            fpn_blob = model.ConvNd(
                fpn_blob, 'fpn_' + str(i),
                dim_in, fpn_dim,
                [cfg.VIDEO.TIME_KERNEL_DIM.BODY, 3, 3],
                pads=2 * [cfg.VIDEO.TIME_KERNEL_DIM.BODY // 2, 1, 1],
                strides=[1, 2, 2],
                weight_init=('XavierFill', {}),
                bias_init=('ConstantFill', {'value': 0.}))
            dim_in = fpn_dim
            blobs_fpn.insert(0, fpn_blob)
            spatial_scales.insert(0, spatial_scales[0] * 0.5)

    return blobs_fpn, fpn_dim, spatial_scales


def add_topdown_lateral_module(
        model, fpn_top, fpn_lateral, fpn_bottom, dim_top, dim_lateral):
    # Lateral 1x1 conv
    lat = model.ConvNd(
        fpn_lateral,
        fpn_bottom if cfg.FPN.INPLACE_LATERAL else fpn_bottom + '_lateral',
        dim_lateral, dim_top, [1, 1, 1],
        pads=2 * [0, 0, 0], strides=[1, 1, 1],
        weight_init=(
            ('ConstantFill', {'value': 0.}) if cfg.FPN.ZERO_INIT_LATERAL
            else ('XavierFill', {})),
        bias_init=('ConstantFill', {'value': 0.}))
    # Upsample in time first
    if cfg.VIDEO.TIME_STRIDE_ON:
        raise NotImplementedError(
            'Need to implement Upsample op in time, and also the time '
            'dimension tend to be like 9->5->3->2 for 9 frame input, '
            'making simple 2x upsample hard. So needs more thought here.')
        # fpn_top = model.Tile(
        #     fpn_top, fpn_top + '_timeUpsample', tiles=2, axis=2)
    # NxCxTxHxW -> (NT)xCxHxW
    temporal_dim = model.GetTemporalDim(fpn_top)
    fpn_top_time2batch = model.MoveTimeToChannelDim(
        fpn_top, fpn_top + '_time2ch')
    # Top-down 2x upsampling
    temp = model.net.UpsampleNearest(
        fpn_top_time2batch, fpn_bottom + '_topdown_timeInCh', scale=2)
    # TODO(rgirdhar): Test and move to the standard C2 op:
    # temp = model.ResizeNearest(
    #     fpn_top_time2batch, fpn_bottom + '_topdown_timeInCh',
    #     width_scale=2.0, height_scale=2.0)
    # Reshape back to NxCxTxHxW
    td = model.MoveTimeToChannelDimInverse(
        temp, fpn_bottom + '_topdown',
        temporal_dim=temporal_dim)
    # Sum lateral and top-down
    model.net.Sum([lat, td], fpn_bottom)


def add_fpn_rpn_outputs(model, blobs_in, dim_in, spatial_scales, time_dim):
    num_anchors = len(cfg.FPN.RPN_ASPECT_RATIOS)
    dim_out = dim_in
    raise NotImplementedError('Redo bbox_targets like in model_builder.py')
    if cfg.VIDEO.DEBUG_USE_RPN_GT:
        raise NotImplementedError('Need to implement this similar to non-FPN')

    k_max = cfg.FPN.RPN_MAX_LEVEL  # coarsest level of pyramid
    k_min = cfg.FPN.RPN_MIN_LEVEL  # finest level of pyramid
    assert len(blobs_in) == k_max - k_min + 1
    for lvl in range(k_min, k_max + 1):
        bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
        sc = spatial_scales[k_max - lvl]  # in reversed order
        slvl = str(lvl)

        if lvl == k_min:
            # Create conv ops with randomly initialized weights and
            # zeroed biases for the first FPN level; these will be shared by
            # all other FPN levels
            # RPN hidden representation
            conv_rpn_fpn = model.ConvNd(
                bl_in, 'conv_rpn_fpn' + slvl,
                dim_in, dim_out, [cfg.VIDEO.TIME_KERNEL_DIM.HEAD_RPN, 3, 3],
                pads=2 * [cfg.VIDEO.TIME_KERNEL_DIM.HEAD_RPN // 2, 1, 1],
                strides=[1, 1, 1],
                weight_init=('GaussianFill', {'std': 0.01}),
                bias_init=('ConstantFill', {'value': 0.}))
            model.Relu(conv_rpn_fpn, conv_rpn_fpn)
            # Convert to 2D. Earlier was averaging in time, now moving to channel
            conv_rpn_fpn_timepool = model.MoveTimeToChannelDim(
                conv_rpn_fpn, 'conv_rpn_timepooled_fpn' + slvl)
            # Proposal classification scores
            rpn_cls_logits_fpn = model.Conv(
                conv_rpn_fpn_timepool, 'rpn_cls_logits_fpn' + slvl,
                dim_out * time_dim, num_anchors, 1, pads=0, stride=1,
                weight_init=('GaussianFill', {'std': 0.01}),
                bias_init=('ConstantFill', {'value': 0.}))
            # Proposal bbox regression deltas
            rpn_bbox_pred_fpn = model.Conv(
                conv_rpn_fpn_timepool, 'rpn_bbox_pred_fpn' + slvl,
                dim_out * time_dim, 4 * time_dim * num_anchors, 1, pad=0, stride=1,
                weight_init=('GaussianFill', {'std': 0.01}),
                bias_init=('ConstantFill', {'value': 0.}))
            # Proposal visibility classification scores
            # TODO(rgirdhar): Need to use this in future
            # rpn_vis_cls_logits_fpn =
            model.Conv(
                conv_rpn_fpn_timepool, 'rpn_vis_cls_logits_fpn' + slvl,
                dim_out * time_dim, num_anchors * time_dim, 1, pads=0, stride=1,
                weight_init=('GaussianFill', {'std': 0.01}),
                bias_init=('ConstantFill', {'value': 0.}))
        else:
            # Share weights and biases
            sk_min = str(k_min)
            # RPN hidden representation
            conv_rpn_fpn = model.ConvShared(
                bl_in, 'conv_rpn_fpn' + slvl,
                dim_in, dim_out, [3, 3, 3],
                pads=2 * [1, 1, 1], strides=[1, 1, 1],
                nd=True,
                weight='conv_rpn_fpn' + sk_min + '_w',
                bias='conv_rpn_fpn' + sk_min + '_b')
            model.Relu(conv_rpn_fpn, conv_rpn_fpn)
            # Convert to 2D. Earlier was averaging in time, now moving to channel
            conv_rpn_fpn_timepool = model.MoveTimeToChannelDim(
                conv_rpn_fpn, 'conv_rpn_timepooled_fpn' + slvl)
            # Proposal classification scores
            rpn_cls_logits_fpn = model.ConvShared(
                conv_rpn_fpn_timepool, 'rpn_cls_logits_fpn' + slvl,
                dim_out * time_dim, num_anchors, 1, pad=0, stride=1,
                weight='rpn_cls_logits_fpn' + sk_min + '_w',
                bias='rpn_cls_logits_fpn' + sk_min + '_b')
            # Proposal bbox regression deltas
            rpn_bbox_pred_fpn = model.ConvShared(
                conv_rpn_fpn_timepool, 'rpn_bbox_pred_fpn' + slvl,
                dim_out * time_dim, 4 * time_dim * num_anchors, 1, pad=0, stride=1,
                weight='rpn_bbox_pred_fpn' + sk_min + '_w',
                bias='rpn_bbox_pred_fpn' + sk_min + '_b')
            # Proposal visibility classification scores
            # TODO(rgirdhar): Need to use this in future
            # rpn_vis_cls_logits_fpn =
            model.ConvShared(
                conv_rpn_fpn_timepool, 'rpn_vis_cls_logits_fpn' + slvl,
                dim_out * time_dim, num_anchors * time_dim, 1, pad=0, stride=1,
                weight='rpn_vis_cls_logits_fpn' + sk_min + '_w',
                bias='rpn_vis_cls_logits_fpn' + sk_min + '_b')

        if not model.train or cfg.MODEL.FASTER_RCNN:
            # Add op that generates RPN proposals
            # The proposals are needed when generating proposals from an
            # RPN-only model at inference time, but *not* when training it
            lvl_anchors = generate_anchors(
                stride=2. ** lvl,
                sizes=(cfg.FPN.RPN_ANCHOR_START_SIZE * 2. ** (lvl - k_min), ),
                aspect_ratios=cfg.FPN.RPN_ASPECT_RATIOS,
                time_dim=time_dim)
            rpn_cls_probs_fpn = model.net.Sigmoid(
                rpn_cls_logits_fpn, 'rpn_cls_probs_fpn' + slvl)
            # Need to use this in future
            # rpn_vis_cls_probs_fpn = model.net.Sigmoid(
            #     rpn_cls_logits_fpn, 'rpn_vis_cls_probs_fpn' + slvl)
            model.GenerateProposals(
                [rpn_cls_probs_fpn, rpn_bbox_pred_fpn, 'im_info'],
                ['rpn_rois_fpn' + slvl, 'rpn_roi_probs_fpn' + slvl],
                anchors=lvl_anchors,
                spatial_scale=sc)
