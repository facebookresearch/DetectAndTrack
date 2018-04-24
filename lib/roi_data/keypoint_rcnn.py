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
import utils.keypoints as keypoint_utils
import utils.blob as blob_utils

import logging
logger = logging.getLogger(__name__)


def add_keypoint_rcnn_blobs(
        blobs, roidb, fg_rois_per_image, fg_inds, im_scale, batch_idx):
    # Note: gt_inds must match how they're computed in
    # datasets.json_dataset._merge_proposal_boxes_into_roidb
    gt_inds = np.where(roidb['gt_classes'] > 0)[0]
    max_overlaps = roidb['max_overlaps']
    gt_keypoints = roidb['gt_keypoints']

    ind_kp = gt_inds[roidb['box_to_gt_ind_map']]
    within_box = _within_box(gt_keypoints[ind_kp, :, :], roidb['boxes'])
    vis_kp = gt_keypoints[ind_kp, 2, :] > 0
    is_visible = np.sum(np.logical_and(vis_kp, within_box), axis=1) > 0
    kp_fg_inds = np.where(
        np.logical_and(max_overlaps >= cfg.TRAIN.FG_THRESH, is_visible))[0]

    kp_fg_rois_per_this_image = np.minimum(
        fg_rois_per_image, kp_fg_inds.size)
    if kp_fg_inds.size > kp_fg_rois_per_this_image:
        kp_fg_inds = np.random.choice(
            kp_fg_inds, size=kp_fg_rois_per_this_image, replace=False)

    if kp_fg_inds.shape[0] == 0:
        kp_fg_inds = gt_inds
    sampled_fg_rois = roidb['boxes'][kp_fg_inds]
    box_to_gt_ind_map = roidb['box_to_gt_ind_map'][kp_fg_inds]

    num_keypoints = gt_keypoints.shape[-1]
    sampled_keypoints = -np.ones(
        (len(sampled_fg_rois), gt_keypoints.shape[1], num_keypoints),
        dtype=gt_keypoints.dtype)
    for ii in range(len(sampled_fg_rois)):
        ind = box_to_gt_ind_map[ii]
        if ind >= 0:
            sampled_keypoints[ii, :, :] = gt_keypoints[gt_inds[ind], :, :]
            # assert np.sum(sampled_keypoints[ii, 2, :]) > 0

    all_heats = []
    all_weights = []
    time_dim = sampled_fg_rois.shape[-1] // 4
    per_frame_nkps = num_keypoints // time_dim
    for t in range(time_dim):
        heats, weights = keypoint_utils.keypoints_to_heatmap_labels(
            sampled_keypoints[..., t * per_frame_nkps: (t + 1) * per_frame_nkps],
            sampled_fg_rois[..., t * 4: (t + 1) * 4])
        all_heats.append(heats)
        all_weights.append(weights)
    heats = np.concatenate(all_heats, axis=-1)
    weights = np.concatenate(all_weights, axis=-1)

    shape = (sampled_fg_rois.shape[0] * cfg.KRCNN.NUM_KEYPOINTS * time_dim, 1)
    heats = heats.reshape(shape)
    weights = weights.reshape(shape)

    sampled_fg_rois *= im_scale
    repeated_batch_idx = batch_idx * blob_utils.ones(
        (sampled_fg_rois.shape[0], 1))
    sampled_fg_rois = np.hstack((repeated_batch_idx, sampled_fg_rois))

    blobs['keypoint_rois'] = sampled_fg_rois
    blobs['keypoint_locations_int32'] = heats.astype(np.int32, copy=False)
    blobs['keypoint_weights'] = weights


def _within_box(points, boxes):
    """
    points : Nx2xK
    boxes: Nx4
    output: NxK
    """
    x_within = np.logical_and(
        points[:, 0, :] >= np.expand_dims(boxes[:, 0], axis=1),
        points[:, 0, :] <= np.expand_dims(boxes[:, 2], axis=1))
    y_within = np.logical_and(
        points[:, 1, :] >= np.expand_dims(boxes[:, 1], axis=1),
        points[:, 1, :] <= np.expand_dims(boxes[:, 3], axis=1))
    return np.logical_and(x_within, y_within)
