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
from collections import namedtuple
import threading

from core.config import cfg
import utils.boxes as box_utils
import utils.blob as blob_utils
from modeling.generate_anchors import generate_anchors, time_extend_shifts

import logging
logger = logging.getLogger(__name__)
npr.seed(cfg.RNG_SEED)


def get_rpn_blob_names(is_training=True):
    # im_info: (height, width, image scale)
    blob_names = ['im_info']
    if is_training:
        # gt boxes: (batch_idx, x1, y1, x2, y2, cls)
        blob_names += ['roidb']
        if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_RPN:
            # Same format as RPN blobs, but one per FPN level
            for lvl in range(cfg.FPN.RPN_MIN_LEVEL, cfg.FPN.RPN_MAX_LEVEL + 1):
                blob_names += [
                    'rpn_labels_int32_wide_fpn' + str(lvl),
                    'rpn_bbox_targets_wide_fpn' + str(lvl),
                    'rpn_bbox_inside_weights_wide_fpn' + str(lvl),
                    'rpn_bbox_outside_weights_wide_fpn' + str(lvl),
                    'rpn_vis_labels_int32_wide_fpn' + str(lvl)]
        else:
            # Single level RPN blobs
            blob_names += [
                'rpn_labels_int32_wide',
                'rpn_bbox_targets_wide',
                'rpn_bbox_inside_weights_wide',
                'rpn_bbox_outside_weights_wide',
                'rpn_vis_labels_int32_wide']
    return blob_names


def _compute_rpn_blobs(im_height, im_width, gt_rois, visible_tracks,
                       all_anchors, foas, foa, k_min, k_max):
    """ Compute RPN blobs on a single image/frame."""
    blobs = {}
    # Add RPN targets
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_RPN:
        # RPN applied to many feature levels, as in the FPN paper
        rpn_blobs = _get_rpn_blobs(
            im_height, im_width, foas, all_anchors, gt_rois, visible_tracks)
        for i, lvl in enumerate(range(k_min, k_max + 1)):
            for k, v in rpn_blobs[i].items():
                blob_name = k + '_fpn' + str(lvl)
                blobs[blob_name] = v
    else:
        # Classical RPN, applied to a single feature level
        rpn_blobs = _get_rpn_blobs(
            im_height, im_width, [foa], all_anchors, gt_rois, visible_tracks)
        for k, v in rpn_blobs.items():
            blobs[k] = v
    return blobs


def _populate_rpn_blobs(entry, scale, blobs, all_anchors, foas, foa, k_min, k_max):
    """ Compute RPN blobs for the whole video by running it on each frame. """
    # If this is a video, need to do the same thing on all frames
    im_height = np.round(entry['height'] * scale)
    im_width = np.round(entry['width'] * scale)
    gt_inds = np.where(
        (entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
    gt_rois = entry['boxes'][gt_inds, :] * scale
    if 'track_visible' in entry:
        visible_tracks = entry['track_visible'][gt_inds, ...]
    else:
        # Non video case, T = 1
        visible_tracks = np.full((gt_inds.shape[0], 1), True)
    local_blobs = _compute_rpn_blobs(
        im_height, im_width, gt_rois, visible_tracks, all_anchors, foas, foa,
        k_min, k_max)
    im_info = np.array([[im_height, im_width, scale]], dtype=np.float32)
    blobs['im_info'].append(im_info)
    for blob_name in local_blobs.keys():
        blobs[blob_name].append(local_blobs[blob_name])


def add_rpn_blobs(blobs, im_scales, roidb):
    """Add blobs needed training RPN-only and end-to-end Faster R-CNN models."""
    # Temporal dimensions of the output
    T = roidb[0]['boxes'].shape[-1] // 4
    # Following vars are only used in FPN case, but keeping it out of the "if"
    # condition, so as to allow for _populate_rpn_blobs to work (it will pass
    # these dummy values and not use them)
    foas = []
    k_max = cfg.FPN.RPN_MAX_LEVEL
    k_min = cfg.FPN.RPN_MIN_LEVEL
    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_RPN:
        # RPN applied to many feature levels, as in the FPN paper
        for lvl in range(k_min, k_max + 1):
            field_stride = 2. ** lvl
            anchor_sizes = (
                cfg.FPN.RPN_ANCHOR_START_SIZE * 2. ** (lvl - k_min), )
            anchor_aspect_ratios = cfg.FPN.RPN_ASPECT_RATIOS
            foa = _get_field_of_anchors(
                field_stride, anchor_sizes, anchor_aspect_ratios, T)
            foas.append(foa)
        all_anchors = np.concatenate([f.field_of_anchors for f in foas])
    else:
        foa = _get_field_of_anchors(
            cfg.RPN.STRIDE, cfg.RPN.SIZES, cfg.RPN.ASPECT_RATIOS, T)
        all_anchors = foa.field_of_anchors

    for im_i, entry in enumerate(roidb):
        _populate_rpn_blobs(entry, im_scales[im_i], blobs, all_anchors, foas,
                            foa, k_min, k_max)

    for k, v in blobs.items():
        if isinstance(v, list) and len(v) > 0:
            blobs[k] = np.concatenate(v)

    valid_keys = [
        'has_visible_keypoints', 'boxes', 'segms', 'seg_areas', 'gt_classes',
        'gt_overlaps', 'is_crowd', 'box_to_gt_ind_map', 'gt_keypoints']
    minimal_roidb = [{} for _ in range(len(roidb))]
    for i, e in enumerate(roidb):
        for k in valid_keys:
            if k in e:
                minimal_roidb[i][k] = e[k]
    blobs['roidb'] = blob_utils.serialize(minimal_roidb)

    # Always return valid=True, since RPN minibatches are valid by design
    return True


FieldOfAnchors = namedtuple(
    'FieldOfAnchors',
    ['field_of_anchors', 'num_cell_anchors', 'stride', 'field_size'])


# Cache for memoizing _get_field_of_anchors
_threadlocal_foa = threading.local()


def _get_field_of_anchors(stride, anchor_sizes, anchor_aspect_ratios, time_dim):
    global _threadlocal_foa
    if not hasattr(_threadlocal_foa, 'cache'):
        _threadlocal_foa.cache = {}

    cache_key = str(stride) + str(anchor_sizes) + str(anchor_aspect_ratios)
    if cache_key in _threadlocal_foa.cache:
        return _threadlocal_foa.cache[cache_key]

    # Anchors at a single feature cell
    cell_anchors = generate_anchors(
        stride=stride,
        sizes=anchor_sizes,
        aspect_ratios=anchor_aspect_ratios,
        time_dim=time_dim)
    num_cell_anchors = cell_anchors.shape[0]

    # Generate canonical proposals from shifted anchors
    # Enumerate all shifted positions on the (H, W) grid
    fpn_max_size = cfg.FPN.COARSEST_STRIDE * np.ceil(
        cfg.TRAIN.MAX_SIZE / float(cfg.FPN.COARSEST_STRIDE))
    field_size = int(np.ceil(fpn_max_size / float(stride)))
    shifts = np.arange(0, field_size) * stride
    shift_x, shift_y = np.meshgrid(shifts, shifts)
    shift_x = shift_x.flatten()
    shift_y = shift_y.flatten()
    shifts = np.vstack((shift_x, shift_y, shift_x, shift_y)).transpose()
    # For the time case, replicate the shifts for other boxes in the tube
    shifts = time_extend_shifts(shifts, time_dim)

    # Broacast anchors over shifts to enumerate all anchors at all positions
    # in the (H, W) grid:
    #   - add A cell anchors of shape (1, A, 4) to
    #   - K shifts of shape (K, 1, 4) to get
    #   - all shifted anchors of shape (K, A, 4)
    #   - reshape to (K*A, 4) shifted anchors
    A = num_cell_anchors
    K = shifts.shape[0]
    field_of_anchors = (
        cell_anchors.reshape((1, A, 4 * time_dim)) +
        shifts.reshape((1, K, 4 * time_dim)).transpose((1, 0, 2)))
    field_of_anchors = field_of_anchors.reshape((K * A, 4 * time_dim))
    foa = FieldOfAnchors(
        field_of_anchors=field_of_anchors.astype(np.float32),
        num_cell_anchors=num_cell_anchors,
        stride=stride,
        field_size=field_size)
    _threadlocal_foa.cache[cache_key] = foa
    return foa


def _get_rpn_blobs(im_height, im_width, foas, all_anchors, gt_boxes,
                   visible_tracks):
    total_anchors = all_anchors.shape[0]
    straddle_thresh = cfg.TRAIN.RPN_STRADDLE_THRESH

    time_dim = all_anchors.shape[1] // 4
    if straddle_thresh >= 0:
        # Only keep anchors inside the image by a margin of straddle_thresh
        # Set TRAIN.RPN_STRADDLE_THRESH to -1 (or a large value) to keep all
        # anchors
        # Using ::4 and np.all to make sure all the boxes in the tube
        # satisfy the conditions
        inds_inside = np.where(
            np.all(all_anchors[:, 0::4] >= -straddle_thresh, axis=1) &
            np.all(all_anchors[:, 1::4] >= -straddle_thresh, axis=1) &
            np.all(all_anchors[:, 2::4] < im_width + straddle_thresh, axis=1) &
            np.all(all_anchors[:, 3::4] < im_height + straddle_thresh, axis=1)
        )[0]
        # keep only inside anchors
        anchors = all_anchors[inds_inside, :]
    else:
        inds_inside = np.arange(all_anchors.shape[0])
        anchors = all_anchors
    num_inside = len(inds_inside)

    logger.debug('total_anchors: {}'.format(total_anchors))
    logger.debug('inds_inside: {}'.format(num_inside))
    logger.debug('anchors.shape: {}'.format(anchors.shape))

    # Compute anchor labels:
    # label=1 is positive, 0 is negative, -1 is don't care (ignore)
    labels = np.empty((num_inside, ), dtype=np.int32)
    labels.fill(-1)
    if len(gt_boxes) > 0:
        # Compute overlaps between the anchors and the gt boxes overlaps
        anchor_by_gt_overlap = box_utils.bbox_overlaps(anchors, gt_boxes)
        # Map from anchor to gt box that has highest overlap
        anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(axis=1)
        # For each anchor, amount of overlap with most overlapping gt box
        anchor_to_gt_max = anchor_by_gt_overlap[
            np.arange(num_inside), anchor_to_gt_argmax]

        # Map from gt box to an anchor that has highest overlap
        gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(axis=0)
        # For each gt box, amount of overlap with most overlapping anchor
        gt_to_anchor_max = anchor_by_gt_overlap[
            gt_to_anchor_argmax, np.arange(anchor_by_gt_overlap.shape[1])]
        # Find all anchors that share the max overlap amount
        # (this includes many ties)
        anchors_with_max_overlap = np.where(
            anchor_by_gt_overlap == gt_to_anchor_max)[0]

        # Fg label: for each gt use anchors with highest overlap
        # (including ties)
        labels[anchors_with_max_overlap] = 1
        # Fg label: above threshold IOU
        labels[anchor_to_gt_max >= cfg.TRAIN.RPN_POSITIVE_OVERLAP] = 1

    # subsample positive labels if we have too many
    num_fg = int(cfg.TRAIN.RPN_FG_FRACTION * cfg.TRAIN.RPN_BATCH_SIZE_PER_IM)
    fg_inds = np.where(labels == 1)[0]
    if len(fg_inds) > num_fg:
        disable_inds = npr.choice(
            fg_inds, size=(len(fg_inds) - num_fg), replace=False)
        labels[disable_inds] = -1
    fg_inds = np.where(labels == 1)[0]

    # subsample negative labels if we have too many
    # (samples with replacement, but since the set of bg inds is large most
    # samples will not have repeats)
    num_bg = cfg.TRAIN.RPN_BATCH_SIZE_PER_IM - np.sum(labels == 1)
    bg_inds = np.where(anchor_to_gt_max < cfg.TRAIN.RPN_NEGATIVE_OVERLAP)[0]
    if len(bg_inds) > num_bg:
        enable_inds = bg_inds[npr.randint(len(bg_inds), size=num_bg)]
        labels[enable_inds] = 0
    bg_inds = np.where(labels == 0)[0]

    bbox_targets = np.zeros((num_inside, 4 * time_dim), dtype=np.float32)
    bbox_targets[fg_inds, :] = _compute_targets(
        anchors[fg_inds, :], gt_boxes[anchor_to_gt_argmax[fg_inds], :])

    # Bbox regression loss has the form:
    #   loss(x) = weight_outside * L(weight_inside * x)
    # Inside weights allow us to set zero loss on an element-wise basis
    # Bbox regression is only trained on positive examples so we set their
    # weights to 1.0 (or otherwise if config is different) and 0 otherwise
    bbox_inside_weights = np.zeros((num_inside, 4 * time_dim), dtype=np.float32)
    # bbox_inside_weights[labels == 1, :] = (1.0, 1.0, 1.0, 1.0)
    # Use the track values to set the weights for track boxes that are not
    # visible to 0
    bbox_inside_weights[fg_inds, :] = np.tile(np.expand_dims(visible_tracks[
        anchor_to_gt_argmax[fg_inds]], -1),
        (1, 1, 4)).reshape((len(fg_inds), -1)).astype(np.float32)

    # The bbox regression loss only averages by the number of images in the
    # mini-batch, whereas we need to average by the total number of example
    # anchors selected
    # Outside weights are used to scale each element-wise loss so the final
    # average over the mini-batch is correct
    bbox_outside_weights = np.zeros((num_inside, 4 * time_dim), dtype=np.float32)
    # uniform weighting of examples (given non-uniform sampling)
    num_examples = np.sum(labels >= 0)
    bbox_outside_weights[labels == 1, :] = 1.0 / num_examples
    bbox_outside_weights[labels == 0, :] = 1.0 / num_examples

    # The tube case also requires predicting if the specific box in the tube
    # is visible or not. Create a copy of labels with the number of time dims
    vis_labels = np.tile(np.expand_dims(labels, -1), [1, time_dim])
    vis_labels[fg_inds, :] *= visible_tracks[
        anchor_to_gt_argmax[fg_inds]].astype(np.int32)

    # Map up to original set of anchors
    labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
    bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
    bbox_inside_weights = _unmap(
        bbox_inside_weights, total_anchors, inds_inside, fill=0)
    bbox_outside_weights = _unmap(
        bbox_outside_weights, total_anchors, inds_inside, fill=0)
    vis_labels = _unmap(vis_labels, total_anchors, inds_inside, fill=-1)

    # Split the generated labels, etc. into labels per each field of anchors
    blobs_out = []
    start_idx = 0
    for foa in foas:
        H = foa.field_size
        W = foa.field_size
        A = foa.num_cell_anchors
        end_idx = start_idx + H * W * A
        _labels = labels[start_idx:end_idx]
        _bbox_targets = bbox_targets[start_idx:end_idx, :]
        _bbox_inside_weights = bbox_inside_weights[start_idx:end_idx, :]
        _bbox_outside_weights = bbox_outside_weights[start_idx:end_idx, :]
        _vis_labels = vis_labels[start_idx:end_idx, :]
        start_idx = end_idx

        # labels output with shape (1, A, height, width)
        _labels = _labels.reshape((1, H, W, A)).transpose(0, 3, 1, 2)
        # bbox_targets output with shape (1, 4T * A, height, width)
        _bbox_targets = _bbox_targets.reshape(
            (1, H, W, A * 4 * time_dim)).transpose(0, 3, 1, 2)
        # bbox_inside_weights output with shape (1, 4T * A, height, width)
        _bbox_inside_weights = _bbox_inside_weights.reshape(
            (1, H, W, A * 4 * time_dim)).transpose(0, 3, 1, 2)
        # bbox_outside_weights output with shape (1, 4T * A, height, width)
        _bbox_outside_weights = _bbox_outside_weights.reshape(
            (1, H, W, A * 4 * time_dim)).transpose(0, 3, 1, 2)
        # vis_labels output with shape (1, T * A, height, width)
        _vis_labels = _vis_labels.reshape(
            (1, H, W, time_dim * A)).transpose(0, 3, 1, 2)
        blobs_out.append(
            dict(
                rpn_labels_int32_wide=_labels,
                rpn_bbox_targets_wide=_bbox_targets,
                rpn_bbox_inside_weights_wide=_bbox_inside_weights,
                rpn_bbox_outside_weights_wide=_bbox_outside_weights,
                rpn_vis_labels_int32_wide=_vis_labels))
    return blobs_out[0] if len(blobs_out) == 1 else blobs_out


def _unmap(data, count, inds, fill=0):
    """Unmap a subset of item (data) back to the original set of items (of
    size count)"""
    if count == len(inds):
        return data

    if len(data.shape) == 1:
        ret = np.empty((count, ), dtype=data.dtype)
        ret.fill(fill)
        ret[inds] = data
    else:
        ret = np.empty((count, ) + data.shape[1:], dtype=data.dtype)
        ret.fill(fill)
        ret[inds, :] = data
    return ret


def _compute_targets(ex_rois, gt_rois):
    """Compute bounding-box regression targets for an image."""
    return box_utils.bbox_transform_inv(
        ex_rois, gt_rois, (1.0, 1.0, 1.0, 1.0)
    ).astype(np.float32, copy=False)
