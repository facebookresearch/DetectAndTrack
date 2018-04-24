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

from core.config import cfg
from datasets.json_dataset import JsonDataset
import utils.boxes as box_utils
import utils.segms as segm_utils
import utils.keypoints as keypoint_utils
import utils.video as video_utils

import logging
logger = logging.getLogger(__name__)


def combined_roidb_for_training(dataset_names, proposal_files):
    def get_roidb(dataset_name, proposal_file):
        ds = JsonDataset(dataset_name)
        roidb = ds.get_roidb(
            gt=True, proposal_file=proposal_file,
            crowd_filter_thresh=cfg.TRAIN.CROWD_FILTER_THRESH)
        if cfg.TRAIN.USE_FLIPPED:
            logger.info('Appending horizontally-flipped training examples...')
            extend_with_flipped_entries(roidb, ds)
        logger.info('Loaded dataset: {:s}'.format(ds.name))
        return roidb

    dataset_names = dataset_names.split(':')
    if proposal_files != '':
        proposal_files = proposal_files.split(':')
    else:
        proposal_files = [None] * len(dataset_names)
    assert len(dataset_names) == len(proposal_files)
    roidbs = [get_roidb(*args) for args in zip(dataset_names, proposal_files)]
    roidb = roidbs[0]
    for r in roidbs[1:]:
        roidb.extend(r)
    roidb = filter_for_training(roidb)

    # Video processing
    if cfg.MODEL.VIDEO_ON:
        roidb = video_utils.get_clip(roidb, remove_imperfect=True)

    if cfg.TRAIN.BBOX_REG:
        logger.info('Computing bounding-box regression targets...')
        add_bbox_regression_targets(roidb)
        logger.info('done')

    _compute_and_log_stats(roidb)

    return roidb


def extend_with_flipped_entries(roidb, dataset):
    flipped_roidb = []
    for entry in roidb:
        width = entry['width']
        boxes = entry['boxes'].copy()
        oldx1 = boxes[:, 0].copy()
        oldx2 = boxes[:, 2].copy()
        boxes[:, 0] = width - oldx2 - 1
        boxes[:, 2] = width - oldx1 - 1
        assert (boxes[:, 2] >= boxes[:, 0]).all()
        flipped_entry = {}
        dont_copy = ('boxes', 'segms', 'gt_keypoints', 'flipped')
        for k, v in entry.items():
            if k not in dont_copy:
                flipped_entry[k] = v
        flipped_entry['boxes'] = boxes
        flipped_entry['segms'] = segm_utils.flip_segms(
            entry['segms'], entry['width'])
        if dataset.keypoints is not None:
            flipped_entry['gt_keypoints'] = keypoint_utils.flip_keypoints(
                dataset.keypoints, dataset.keypoint_flip_map,
                entry['gt_keypoints'], entry['width'])
        flipped_entry['flipped'] = True
        flipped_roidb.append(flipped_entry)
    roidb.extend(flipped_roidb)


def filter_for_training(roidb):
    """Remove roidb entries that have no usable RoIs based on config settings.
    """
    def is_valid(entry):
        # Valid images have:
        #   (1) At least one foreground RoI OR
        #   (2) At least one background RoI
        overlaps = entry['max_overlaps']
        # find boxes with sufficient overlap
        fg_inds = np.where(overlaps >= cfg.TRAIN.FG_THRESH)[0]
        # Select background RoIs as those within [BG_THRESH_LO, BG_THRESH_HI)
        bg_inds = np.where((overlaps < cfg.TRAIN.BG_THRESH_HI) &
                           (overlaps >= cfg.TRAIN.BG_THRESH_LO))[0]
        # image is only valid if such boxes exist
        valid = len(fg_inds) > 0 or len(bg_inds) > 0
        if cfg.MODEL.KEYPOINTS_ON:
            # If we're training for keypoints, exclude images with no keypoints
            valid = valid and entry['has_visible_keypoints']
        return valid

    num = len(roidb)
    filtered_roidb = [entry for entry in roidb if is_valid(entry)]
    num_after = len(filtered_roidb)
    logger.info('Filtered {} roidb entries: {} -> {}'.
                format(num - num_after, num, num_after))
    return filtered_roidb


def add_bbox_regression_targets(roidb):
    """Add information needed to train bounding-box regressors."""
    for entry in roidb:
        entry['bbox_targets'] = _compute_targets(entry)


def _compute_targets(entry):
    """Compute bounding-box regression targets for an image."""
    # Indices of ground-truth ROIs
    rois = entry['boxes']
    overlaps = entry['max_overlaps']
    labels = entry['max_classes']
    gt_inds = np.where((entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
    # Targets has format (class, tx, ty, tw, th, tx2, ty2, tw2, th2...)
    # (for each time frame)
    targets = np.zeros((rois.shape[0], rois.shape[1] + 1), dtype=np.float32)
    if len(gt_inds) == 0:
        # Bail if the image has no ground-truth ROIs
        return targets

    # Indices of examples for which we try to make predictions
    ex_inds = np.where(overlaps >= cfg.TRAIN.BBOX_THRESH)[0]

    # Get IoU overlap between each ex ROI and gt ROI
    ex_gt_overlaps = box_utils.bbox_overlaps(
        rois[ex_inds, :].astype(dtype=np.float32, copy=False),
        rois[gt_inds, :].astype(dtype=np.float32, copy=False))

    # Find which gt ROI each ex ROI has max overlap with:
    # this will be the ex ROI's gt target
    gt_assignment = ex_gt_overlaps.argmax(axis=1)
    gt_rois = rois[gt_inds[gt_assignment], :]
    ex_rois = rois[ex_inds, :]
    # Use class "1" for all boxes if using class_agnostic_bbox_reg
    targets[ex_inds, 0] = (
        1 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else labels[ex_inds])
    targets[ex_inds, 1:] = box_utils.bbox_transform_inv(
        ex_rois, gt_rois, cfg.MODEL.BBOX_REG_WEIGHTS)
    return targets


def _compute_and_log_stats(roidb):
    classes = roidb[0]['dataset'].classes
    char_len = np.max([len(c) for c in classes])
    hist_bins = np.arange(len(classes) + 1)

    # Histogram of ground-truth objects
    gt_hist = np.zeros((len(classes)), dtype=np.int)
    for entry in roidb:
        gt_inds = np.where(
            (entry['gt_classes'] > 0) & (entry['is_crowd'] == 0))[0]
        gt_classes = entry['gt_classes'][gt_inds]
        gt_hist += np.histogram(gt_classes, bins=hist_bins)[0]
    logger.info('Ground-truth class histogram:')
    for i, v in enumerate(gt_hist):
        logger.info(
            '{:d}{:s}: {:d}'.format(
                i, classes[i].rjust(char_len), v))
    logger.info('-' * char_len)
    logger.info(
        '{:s}: {:d}'.format(
            'total'.rjust(char_len), np.sum(gt_hist)))
