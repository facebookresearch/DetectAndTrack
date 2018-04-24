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
import numpy.random as npr

from core.config import cfg
import utils.blob as blob_utils
import utils.boxes as box_utils
import roi_data.mask_rcnn
import roi_data.keypoint_rcnn
import modeling.FPN as fpn

import logging
logger = logging.getLogger(__name__)
npr.seed(cfg.RNG_SEED)


def get_fast_rcnn_blob_names(is_training=True):
    # rois blob: holds R regions of interest, each is a 5-tuple
    # (batch_idx, x1, y1, x2, y2) specifying an image batch index and a
    # rectangle (x1, y1, x2, y2)
    blob_names = ['rois']
    if is_training:
        # labels_int32 blob: R categorical labels in [0, ..., K] for K
        # foreground classes plus background
        blob_names += ['labels_int32']
    if is_training and cfg.TRAIN.BBOX_REG:
        # bbox_targets blob: R bounding-box regression targets with 4
        # targets per class
        blob_names += ['bbox_targets']
        # bbox_inside_weights blob: At most 4 targets per roi are active
        # this binary vector sepcifies the subset of active targets
        blob_names += ['bbox_inside_weights']
        blob_names += ['bbox_outside_weights']
    if is_training and cfg.MODEL.MASK_ON:
        # 'mask_rois': RoIs sampled for training the mask prediction branch.
        # Shape is (#masks, 5) in format (batch_idx, x1, y1, x2, y2).
        blob_names += ['mask_rois']
        # 'roi_has_mask': binary labels for the RoIs specified in 'rois'
        # indicating if each RoI has a mask or not. Note that in some cases
        # a *bg* RoI will have an all -1 (ignore) mask associated with it in
        # the case that no fg RoIs can be sampled. Shape is (batchsize).
        blob_names += ['roi_has_mask_int32']
        # 'masks_int32' holds binary masks for the RoIs specified in
        # 'mask_rois'. Shape is (#fg, M * M) where M is the ground truth
        # mask size.
        blob_names += ['masks_int32']
    if is_training and cfg.MODEL.KEYPOINTS_ON:
        # 'keypoint_rois': RoIs sampled for training the keypoint prediction
        # branch. Shape is (#instances, 5) in format (batch_idx, x1, y1, x2,
        # y2).
        blob_names += ['keypoint_rois']
        # 'keypoint_locations_int32': index of keypoint in
        # KRCNN.HEATMAP_SIZE**2 sized array. Shape is (#instances). Used in
        # SoftmaxWithLoss.
        blob_names += ['keypoint_locations_int32']
        # 'keypoint_weights': weight assigned to each target in
        # 'keypoint_locations_int32'. Shape is (#instances). Used in
        # SoftmaxWithLoss.
        blob_names += ['keypoint_weights']
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        # Support for FPN multi-level rois without bbox reg isn't
        # implemented (... and may never be implemented)
        assert cfg.TRAIN.BBOX_REG, \
            'Bounding box regression must be on for FPN multi-level RoIs'
        k_max = cfg.FPN.ROI_MAX_LEVEL
        k_min = cfg.FPN.ROI_MIN_LEVEL
        # Same format as rois blob, but one per FPN level
        for lvl in range(k_min, k_max + 1):
            blob_names += ['rois_fpn' + str(lvl)]
        blob_names += ['rois_idx_restore_int32']
        if is_training:
            if cfg.MODEL.MASK_ON:
                for lvl in range(k_min, k_max + 1):
                    blob_names += ['mask_rois_fpn' + str(lvl)]
                blob_names += ['mask_rois_idx_restore_int32']
            if cfg.MODEL.KEYPOINTS_ON:
                for lvl in range(k_min, k_max + 1):
                    blob_names += ['keypoint_rois_fpn' + str(lvl)]
                blob_names += ['keypoint_rois_idx_restore_int32']
    return blob_names


def add_fast_rcnn_blobs(blobs, im_scales, roidb):
    """Add blobs needed for training Fast R-CNN style models."""
    # Sample training RoIs from each image and append them to the blob lists
    for im_i, entry in enumerate(roidb):
        frcn_blobs = _sample_rois(entry, im_scales[im_i], im_i)
        for k, v in frcn_blobs.items():
            blobs[k].append(v)
    # Concat the training blob lists into tensors
    for k, v in blobs.items():
        if isinstance(v, list) and len(v) > 0:
            blobs[k] = np.concatenate(v)
    # Add FPN multilevel training RoIs, if configured
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois(blobs)

    valid = True
    if cfg.MODEL.KEYPOINTS_ON:
        min_count = cfg.KRCNN.MIN_KEYPOINT_COUNT_FOR_VALID_MINIBATCH
        valid = (
            valid and len(blobs['keypoint_weights']) > 0 and
            np.sum(blobs['keypoint_weights']) > min_count)
    return valid


def _sample_rois(roidb, im_scale, batch_idx):
    """Generate a random sample of RoIs comprising foreground and background
    examples.
    """
    rois_per_image = int(cfg.TRAIN.BATCH_SIZE_PER_IM)
    fg_rois_per_image = int(np.round(cfg.TRAIN.FG_FRACTION * rois_per_image))
    max_overlaps = roidb['max_overlaps']

    # Select foreground RoIs as those with >= FG_THRESH overlap
    fg_inds = np.where(max_overlaps >= cfg.TRAIN.FG_THRESH)[0]
    # Guard against the case when an image has fewer than fg_rois_per_image
    # foreground RoIs
    fg_rois_per_this_image = np.minimum(fg_rois_per_image, fg_inds.size)
    # Sample foreground regions without replacement
    if fg_inds.size > 0:
        fg_inds = npr.choice(
            fg_inds, size=fg_rois_per_this_image, replace=False)

    # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
    bg_inds = np.where((max_overlaps < cfg.TRAIN.BG_THRESH_HI) &
                       (max_overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
    # Compute number of background RoIs to take from this image (guarding
    # against there being fewer than desired)
    bg_rois_per_this_image = rois_per_image - fg_rois_per_this_image
    bg_rois_per_this_image = np.minimum(
        bg_rois_per_this_image, bg_inds.size)
    # Sample foreground regions without replacement
    if bg_inds.size > 0:
        bg_inds = npr.choice(
            bg_inds, size=bg_rois_per_this_image, replace=False)

    # The indices that we're selecting (both fg and bg)
    keep_inds = np.append(fg_inds, bg_inds)
    # Label is the class each RoI has max overlap with
    sampled_labels = roidb['max_classes'][keep_inds]
    sampled_labels[fg_rois_per_this_image:] = 0  # Label bg RoIs with class 0
    sampled_boxes = roidb['boxes'][keep_inds]

    if 'bbox_targets' not in roidb:
        gt_inds = np.where(roidb['gt_classes'] > 0)[0]
        gt_boxes = roidb['boxes'][gt_inds, :]
        gt_assignments = gt_inds[roidb['box_to_gt_ind_map'][keep_inds]]
        bbox_targets = _compute_targets(
            sampled_boxes, gt_boxes[gt_assignments, :], sampled_labels)
        bbox_targets, bbox_inside_weights = _expand_bbox_targets(bbox_targets)
    else:
        bbox_targets, bbox_inside_weights = _expand_bbox_targets(
            roidb['bbox_targets'][keep_inds, :])

    bbox_outside_weights = np.array(
        bbox_inside_weights > 0, dtype=bbox_inside_weights.dtype)

    # Scale rois and format as (batch_idx, x1, y1, x2, y2)
    sampled_rois = sampled_boxes * im_scale
    repeated_batch_idx = batch_idx * blob_utils.ones((sampled_rois.shape[0], 1))
    sampled_rois = np.hstack((repeated_batch_idx, sampled_rois))

    # Base Fast R-CNN blobs
    blob_dict = dict(
        labels_int32=sampled_labels.astype(np.int32, copy=False),
        rois=sampled_rois,
        bbox_targets=bbox_targets,
        bbox_inside_weights=bbox_inside_weights,
        bbox_outside_weights=bbox_outside_weights)

    # Optionally add Mask R-CNN blobs
    if cfg.MODEL.MASK_ON:
        roi_data.mask_rcnn.add_mask_rcnn_blobs(
            blob_dict, sampled_boxes, roidb, im_scale, batch_idx)

    # Optionally add Keypoint R-CNN blobs
    if cfg.MODEL.KEYPOINTS_ON:
        roi_data.keypoint_rcnn.add_keypoint_rcnn_blobs(
            blob_dict, roidb, fg_rois_per_image, fg_inds, im_scale, batch_idx)

    return blob_dict


def _compute_targets(ex_rois, gt_rois, labels):
    """Compute bounding-box regression targets for an image."""

    assert ex_rois.shape[0] == gt_rois.shape[0]
    # Following are no longer true with tubes. Also, since bbox_transform_inv
    # can handle tubes, we don't need these assertions
    # assert ex_rois.shape[1] == 4
    # assert gt_rois.shape[1] == 4

    targets = box_utils.bbox_transform_inv(
        ex_rois, gt_rois, cfg.MODEL.BBOX_REG_WEIGHTS)
    return np.hstack(
        (labels[:, np.newaxis], targets)).astype(np.float32, copy=False)


def _expand_bbox_targets(bbox_target_data):
    """Bounding-box regression targets are stored in a compact form in the
    roidb.

    This function expands those targets into the 4-of-4*K representation used
    by the network (i.e. only one class has non-zero targets). The loss weights
    are similarly expanded.

    Returns:
        bbox_target_data (ndarray): N x 4K blob of regression targets
        bbox_inside_weights (ndarray): N x 4K blob of loss weights
    """
    tube_dim = bbox_target_data.shape[-1] - 1
    num_bbox_reg_classes = cfg.MODEL.NUM_CLASSES
    if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
        num_bbox_reg_classes = 2  # bg and fg

    clss = bbox_target_data[:, 0]
    bbox_targets = blob_utils.zeros((clss.size, tube_dim * num_bbox_reg_classes))
    bbox_inside_weights = blob_utils.zeros(bbox_targets.shape)
    inds = np.where(clss > 0)[0]
    for ind in inds:
        cls = int(clss[ind])
        start = tube_dim * cls
        end = start + tube_dim
        bbox_targets[ind, start:end] = bbox_target_data[ind, 1:]
        bbox_inside_weights[ind, start:end] = 1.0  # (1.0, 1.0, 1.0, 1.0)
    return bbox_targets, bbox_inside_weights


def _add_multilevel_rois(blobs):
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL
    # The map_rois_to_fpn_levels and add_multilevel_roi_blobs functions are
    # shared among the 2D and 3D models.
    lvls = fpn.map_rois_to_fpn_levels(blobs['rois'][:, 1:], lvl_min, lvl_max)
    fpn.add_multilevel_roi_blobs(
        blobs, 'rois', blobs['rois'], lvls, lvl_min, lvl_max)

    if cfg.MODEL.MASK_ON:
        # Masks use the same rois as the box/cls head
        fpn.add_multilevel_roi_blobs(
            blobs, 'mask_rois', blobs['rois'], lvls, lvl_min, lvl_max,
            valid_levels=blobs['roi_has_mask_int32'])

    if cfg.MODEL.KEYPOINTS_ON:
        # Keypoints use a separate set of training rois
        lvls = fpn.map_rois_to_fpn_levels(
            blobs['keypoint_rois'][:, 1:], lvl_min, lvl_max)
        fpn.add_multilevel_roi_blobs(
            blobs, 'keypoint_rois', blobs['keypoint_rois'], lvls, lvl_min,
            lvl_max)
