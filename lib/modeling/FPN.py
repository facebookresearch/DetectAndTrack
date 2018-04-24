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

import numpy as np

import utils.blob as blob_utils
import utils.boxes as box_utils
from core.config import cfg
from modeling.generate_anchors import generate_anchors
import modeling.ResNet as ResNet
from caffe2.python import scope

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
    model.Conv(
        stage_info.blobs[0], 'fpn_inner_' + stage_info.blobs[0],
        dim_in=stage_info.dims[0], dim_out=fpn_dim, kernel=1,
        pad=0, stride=1,
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
        fpn_blob = model.Conv(
            'fpn_inner_' + stage_info.blobs[i], 'fpn_' + stage_info.blobs[i],
            dim_in=fpn_dim, dim_out=fpn_dim, kernel=3,
            pad=1, stride=1,
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
            kernel=1, pad=0, stride=2)
        blobs_fpn.insert(0, P6_blob)
        spatial_scales.insert(0, spatial_scales[0] * 0.5)

    # Newer style from our one-stage detection paper
    if cfg.FPN.EXTRA_CONV_LEVELS and max_level > HIGHEST_LVL:
        fpn_blob = stage_info.blobs[0]
        dim_in = stage_info.dims[0]
        for i in range(HIGHEST_LVL + 1, max_level + 1):
            if i > HIGHEST_LVL + 1:
                fpn_blob = model.Relu(fpn_blob, fpn_blob)
            fpn_blob = model.Conv(
                fpn_blob, 'fpn_' + str(i),
                dim_in=dim_in, dim_out=fpn_dim,
                kernel=3, pad=1, stride=2,
                weight_init=('XavierFill', {}),
                bias_init=('ConstantFill', {'value': 0.}))
            dim_in = fpn_dim
            blobs_fpn.insert(0, fpn_blob)
            spatial_scales.insert(0, spatial_scales[0] * 0.5)

    return blobs_fpn, fpn_dim, spatial_scales


def add_topdown_lateral_module(
        model, fpn_top, fpn_lateral, fpn_bottom, dim_top, dim_lateral):
    # Lateral 1x1 conv
    lat = model.Conv(
        fpn_lateral,
        fpn_bottom if cfg.FPN.INPLACE_LATERAL else fpn_bottom + '_lateral',
        dim_in=dim_lateral, dim_out=dim_top, kernel=1,
        pad=0, stride=1,
        weight_init=(
            ('ConstantFill', {'value': 0.}) if cfg.FPN.ZERO_INIT_LATERAL
            else ('XavierFill', {})),
        bias_init=('ConstantFill', {'value': 0.}))
    # Top-down 2x upsampling
    td = model.net.UpsampleNearest(fpn_top, fpn_bottom + '_topdown', scale=2)
    # Sum lateral and top-down
    model.net.Sum([lat, td], fpn_bottom)


def add_fpn_rpn_outputs(model, blobs_in, dim_in, spatial_scales, time_dim=1):
    # time_dim is only for consistency with 3D function, not used here.
    num_anchors = len(cfg.FPN.RPN_ASPECT_RATIOS)
    dim_out = dim_in

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
            conv_rpn_fpn = model.Conv(
                bl_in, 'conv_rpn_fpn' + slvl,
                dim_in, dim_out, 3, pad=1, stride=1,
                weight_init=('GaussianFill', {'std': 0.01}),
                bias_init=('ConstantFill', {'value': 0.}))
            model.Relu(conv_rpn_fpn, conv_rpn_fpn)
            # Proposal classification scores
            rpn_cls_logits_fpn = model.Conv(
                conv_rpn_fpn, 'rpn_cls_logits_fpn' + slvl,
                dim_in, num_anchors, 1, pad=0, stride=1,
                weight_init=('GaussianFill', {'std': 0.01}),
                bias_init=('ConstantFill', {'value': 0.}))
            # Proposal bbox regression deltas
            rpn_bbox_pred_fpn = model.Conv(
                conv_rpn_fpn, 'rpn_bbox_pred_fpn' + slvl,
                dim_in, 4 * num_anchors, 1, pad=0, stride=1,
                weight_init=('GaussianFill', {'std': 0.01}),
                bias_init=('ConstantFill', {'value': 0.}))
        else:
            # Share weights and biases
            sk_min = str(k_min)
            # RPN hidden representation
            conv_rpn_fpn = model.ConvShared(
                bl_in, 'conv_rpn_fpn' + slvl,
                dim_in, dim_out, 3, pad=1, stride=1,
                weight='conv_rpn_fpn' + sk_min + '_w',
                bias='conv_rpn_fpn' + sk_min + '_b')
            model.Relu(conv_rpn_fpn, conv_rpn_fpn)
            # Proposal classification scores
            rpn_cls_logits_fpn = model.ConvShared(
                conv_rpn_fpn, 'rpn_cls_logits_fpn' + slvl,
                dim_in, num_anchors, 1, pad=0, stride=1,
                weight='rpn_cls_logits_fpn' + sk_min + '_w',
                bias='rpn_cls_logits_fpn' + sk_min + '_b')
            # Proposal bbox regression deltas
            rpn_bbox_pred_fpn = model.ConvShared(
                conv_rpn_fpn, 'rpn_bbox_pred_fpn' + slvl,
                dim_in, 4 * num_anchors, 1, pad=0, stride=1,
                weight='rpn_bbox_pred_fpn' + sk_min + '_w',
                bias='rpn_bbox_pred_fpn' + sk_min + '_b')

        if not model.train or cfg.MODEL.FASTER_RCNN:
            # Add op that generates RPN proposals
            # The proposals are needed when generating proposals from an
            # RPN-only model at inference time, but *not* when training it
            lvl_anchors = generate_anchors(
                stride=2. ** lvl,
                sizes=(cfg.FPN.RPN_ANCHOR_START_SIZE * 2. ** (lvl - k_min), ),
                aspect_ratios=cfg.FPN.RPN_ASPECT_RATIOS,
                time_dim=1)
            rpn_cls_probs_fpn = model.net.Sigmoid(
                rpn_cls_logits_fpn, 'rpn_cls_probs_fpn' + slvl)
            model.GenerateProposals(
                [rpn_cls_probs_fpn, rpn_bbox_pred_fpn, 'im_info'],
                ['rpn_rois_fpn' + slvl, 'rpn_roi_probs_fpn' + slvl],
                anchors=lvl_anchors,
                spatial_scale=sc)


def add_fpn_rpn_losses(model, time_dim=1):
    """ Note that this is shared with FPN3D.py. So this same loss function
    is used with 3D RPN head. """
    loss_gradients = {}
    for lvl in range(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.RPN_MAX_LEVEL + 1):
        slvl = str(lvl)
        # Spatially narrow the full-sized RPN label arrays to match the feature map
        # shape
        model.net.SpatialNarrowAs(
            ['rpn_labels_int32_wide_fpn' + slvl, 'rpn_cls_logits_fpn' + slvl],
            'rpn_labels_int32_fpn' + slvl)
        for key in ('targets', 'inside_weights', 'outside_weights'):
            model.net.SpatialNarrowAs(
                ['rpn_bbox_' + key + '_wide_fpn' + slvl,
                 'rpn_bbox_pred_fpn' + slvl],
                'rpn_bbox_' + key + '_fpn' + slvl)
        loss_rpn_cls_fpn = model.net.SigmoidCrossEntropyLoss(
            ['rpn_cls_logits_fpn' + slvl, 'rpn_labels_int32_fpn' + slvl],
            'loss_rpn_cls_fpn' + slvl,
            normalize=0,
            scale=(1. / cfg.NUM_GPUS / cfg.TRAIN.RPN_BATCH_SIZE_PER_IM /
                   cfg.TRAIN.IMS_PER_BATCH))
        # Normalization by (1) RPN_BATCH_SIZE_PER_IM and (2) IMS_PER_BATCH is
        # handled by (1) setting bbox outside weights and (2) SmoothL1Loss
        # normalizes by IMS_PER_BATCH
        loss_rpn_bbox_fpn = model.net.SmoothL1Loss(
            ['rpn_bbox_pred_fpn' + slvl,
             'rpn_bbox_targets_fpn' + slvl,
             'rpn_bbox_inside_weights_fpn' + slvl,
             'rpn_bbox_outside_weights_fpn' + slvl],
            'loss_rpn_bbox_fpn' + slvl,
            beta=1. / 9.,
            scale=1. / cfg.NUM_GPUS / time_dim)
        loss_gradients.update(
            blob_utils.get_loss_gradients(
                model, [loss_rpn_cls_fpn, loss_rpn_bbox_fpn]))
        model.losses = list(
            set(model.losses +
                ['loss_rpn_cls_fpn' + slvl, 'loss_rpn_bbox_fpn' + slvl]))
    return loss_gradients


def add_fpn_rpn_vis_losses(model):
    """ Note that this is shared with FPN3D.py. So this same loss function
    is used with 3D RPN head. """
    loss_gradients = {}
    for lvl in range(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.RPN_MAX_LEVEL + 1):
        slvl = str(lvl)
        if (model.net.BlobIsDefined(scope.CurrentNameScope() +
                                    'rpn_vis_cls_logits_fpn' + slvl)):
            model.net.SpatialNarrowAs(
                ['rpn_vis_labels_int32_wide_fpn' + slvl,
                 'rpn_vis_cls_logits_fpn' + slvl],
                'rpn_vis_labels_int32_fpn' + slvl)
            loss_rpn_vis_cls_fpn = model.net.SigmoidCrossEntropyLoss(
                ['rpn_vis_cls_logits_fpn' + slvl,
                 'rpn_vis_labels_int32_fpn' + slvl],
                'loss_rpn_vis_cls_fpn' + slvl,
                normalize=0,
                scale=(1. / cfg.NUM_GPUS / cfg.TRAIN.RPN_BATCH_SIZE_PER_IM /
                       cfg.TRAIN.IMS_PER_BATCH))
            loss_gradients.update(
                blob_utils.get_loss_gradients(model, [loss_rpn_vis_cls_fpn]))
            model.losses = list(set(model.losses + ['loss_rpn_vis_cls_fpn' + slvl]))
    return loss_gradients


def map_rois_to_fpn_levels(rois, k_min, k_max):
    # Compute level ids
    s = np.sqrt(box_utils.boxes_area(rois))
    s0 = cfg.FPN.ROI_CANONICAL_SCALE  # default: 224
    lvl0 = cfg.FPN.ROI_CANONICAL_LEVEL  # default: 4

    lvls = np.floor(lvl0 + np.log2(s / s0 + 1e-6))  # Eqn.(1) in FPN paper
    lvls = np.clip(lvls, k_min, k_max)
    # lvls is a list of length len(rois) with a ID from k_min to k_max, as to
    # where it maps to. This might lead to some levels from k_min to k_max not
    # getting any rois mapped to them.
    return lvls


def add_multilevel_roi_blobs(
        blobs, blob_name, rois, lvls, lvl_min, lvl_max, valid_levels=None):
    if valid_levels is None:
        valid_levels = 1
    rois_idx_order = np.empty((0, ))
    rois_stacked = np.zeros((0, rois.shape[-1]), dtype=np.float32)  # for assert
    for lvl in range(lvl_min, lvl_max + 1):
        idx_lvl = np.where(lvls * valid_levels == lvl)[0]
        blobs[blob_name + '_fpn' + str(lvl)] = rois[idx_lvl, :]
        rois_idx_order = np.concatenate((rois_idx_order, idx_lvl))
        rois_stacked = np.vstack(
            [rois_stacked, blobs[blob_name + '_fpn' + str(lvl)]])
    rois_idx_restore = np.argsort(rois_idx_order).astype(np.int32, copy=False)
    blobs[blob_name + '_idx_restore_int32'] = rois_idx_restore
    # Sanity check that restore order is correct
    if isinstance(valid_levels, int) and valid_levels == 1:
        assert (rois_stacked[rois_idx_restore] == rois).all()
    else:
        assert (rois_stacked[rois_idx_restore] == rois[valid_levels == 1]).all()
