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
import cv2

from core.config import cfg
import utils.blob as blob_utils

# OpenCL is enabled by default in OpenCV3 and it is not thread-safe leading
# to huge GPU memory allocations. See https://fburl.com/9d7tvusd
try:
  cv2.ocl.setUseOpenCL(False)
except AttributeError:
  pass


def get_keypoints():
    # Keypoints are not available in the COCO json for the test split, so we
    # provide them here.
    keypoints = [
        'nose',
        'left_eye',
        'right_eye',
        'left_ear',
        'right_ear',
        'left_shoulder',
        'right_shoulder',
        'left_elbow',
        'right_elbow',
        'left_wrist',
        'right_wrist',
        'left_hip',
        'right_hip',
        'left_knee',
        'right_knee',
        'left_ankle',
        'right_ankle']
    keypoint_flip_map = {
        'left_eye': 'right_eye',
        'left_ear': 'right_ear',
        'left_shoulder': 'right_shoulder',
        'left_elbow': 'right_elbow',
        'left_wrist': 'right_wrist',
        'left_hip': 'right_hip',
        'left_knee': 'right_knee',
        'left_ankle': 'right_ankle'}
    return keypoints, keypoint_flip_map


def get_person_class_index():
    return 1


def flip_keypoints(keypoints, keypoint_flip_map, gt_keypoints, width):
    flipped_kps = gt_keypoints.copy()
    for lkp, rkp in keypoint_flip_map.items():
        lid = keypoints.index(lkp)
        rid = keypoints.index(rkp)
        flipped_kps[:, :, lid] = gt_keypoints[:, :, rid]
        flipped_kps[:, :, rid] = gt_keypoints[:, :, lid]

    # Flip x coordinates
    flipped_kps[:, 0, :] = width - flipped_kps[:, 0, :] - 1
    # Maintain COCO convention that if visibility == 0, then x, y = 0
    inds = np.where(flipped_kps[:, 2, :] == 0)
    flipped_kps[inds[0], 0, inds[1]] = 0
    return flipped_kps


def flip_heatmaps(heatmaps):
    """Flips heatmaps horizontally."""
    keypoints, flip_map = get_keypoints()
    heatmaps_flipped = heatmaps.copy()
    for lkp, rkp in flip_map.items():
        lid = keypoints.index(lkp)
        rid = keypoints.index(rkp)
        heatmaps_flipped[:, rid, :, :] = heatmaps[:, lid, :, :]
        heatmaps_flipped[:, lid, :, :] = heatmaps[:, rid, :, :]
    heatmaps_flipped = heatmaps_flipped[:, :, :, ::-1]
    return heatmaps_flipped


def heatmaps_to_keypoints(maps, rois):
    """Extracts predicted keypoint locations from heatmaps. Output has shape
    (#rois, 4, #keypoints) with the 4 rows corresponding to (x, y, logit, prob)
    for each keypoint.
    rgirdhar: Don't modify for tubes, the modification was done in core/test.py
    """
    # This function converts a discrete image coordinate in a HEATMAP_SIZE x
    # HEATMAP_SIZE image to a continuous keypoint coordinate. We maintain
    # consistency with keypoints_to_heatmap_labels by using the conversion from
    # Heckbert 1990: c = d + 0.5, where d is a discrete coordinate and c is a
    # continuous coordinate.
    offset_x = rois[:, 0]
    offset_y = rois[:, 1]

    widths = rois[:, 2] - rois[:, 0]
    heights = rois[:, 3] - rois[:, 1]
    widths = np.maximum(widths, 1)
    heights = np.maximum(heights, 1)
    widths_ceil = np.ceil(widths)
    heights_ceil = np.ceil(heights)

    # NCHW to NHWC for use with OpenCV
    maps = np.transpose(maps, [0, 2, 3, 1])
    min_size = cfg.KRCNN.INFERENCE_MIN_SIZE
    xy_preds = np.zeros(
        (len(rois), 4, cfg.KRCNN.NUM_KEYPOINTS), dtype=np.float32)
    for i in range(len(rois)):
        if min_size > 0:
            roi_map_width = int(np.maximum(widths_ceil[i], min_size))
            roi_map_height = int(np.maximum(heights_ceil[i], min_size))
        else:
            roi_map_width = widths_ceil[i]
            roi_map_height = heights_ceil[i]
        width_correction = widths[i] / roi_map_width
        height_correction = heights[i] / roi_map_height
        roi_map = cv2.resize(
            maps[i], (roi_map_width, roi_map_height),
            interpolation=cv2.INTER_CUBIC)
        # Bring back to CHW
        roi_map = np.transpose(roi_map, [2, 0, 1])
        roi_map_probs = scores_to_probs(roi_map.copy())
        w = roi_map.shape[2]
        for k in range(cfg.KRCNN.NUM_KEYPOINTS):
            pos = roi_map[k, :, :].argmax()
            x_int = pos % w
            y_int = (pos - x_int) // w
            assert (roi_map_probs[k, y_int, x_int] ==
                    roi_map_probs[k, :, :].max())
            x = (x_int + 0.5) * width_correction
            y = (y_int + 0.5) * height_correction
            xy_preds[i, 0, k] = x + offset_x[i]
            xy_preds[i, 1, k] = y + offset_y[i]
            xy_preds[i, 2, k] = roi_map[k, y_int, x_int]
            xy_preds[i, 3, k] = roi_map_probs[k, y_int, x_int]

    return xy_preds


def keypoints_to_heatmap_labels(keypoints, rois):
    """Generate location of heatmap
    Each roi and each keypoint -> xy location of keypoint
    For SoftmaxWithLoss across space
    rgirdhar: Don't modify for tubes, the modification was done in
    roi_data/keypoint_rcnn.py
    """
    # Maps keypoints from the half-open interval [x1, x2) on continuous image
    # coordinates to the closed interval [0, HEATMAP_SIZE - 1] on discrete image
    # coordinates. We use the continuous <-> discrete conversion from Heckbert
    # 1990 ("What is the coordinate of a pixel?"): d = floor(c) and c = d + 0.5,
    # where d is a discrete coordinate and c is a continuous coordinate.
    assert keypoints.shape[2] == cfg.KRCNN.NUM_KEYPOINTS

    shape = (len(rois), cfg.KRCNN.NUM_KEYPOINTS)
    heatmaps = blob_utils.zeros(shape)
    weights = blob_utils.zeros(shape)

    offset_x = rois[:, 0]
    offset_y = rois[:, 1]
    # +1 added by rgirdhar, to avoid divides by 0
    scale_x = cfg.KRCNN.HEATMAP_SIZE / (rois[:, 2] - rois[:, 0] + 1)
    scale_y = cfg.KRCNN.HEATMAP_SIZE / (rois[:, 3] - rois[:, 1] + 1)

    for kp in range(keypoints.shape[2]):
        vis = keypoints[:, 2, kp] > 0
        x = keypoints[:, 0, kp].astype(np.float32)
        y = keypoints[:, 1, kp].astype(np.float32)
        # Since we use floor below, if a keypoint is exactly on the roi's right
        # or bottom boundary, we shift it in by eps (conceptually) to keep it in
        # the ground truth heatmap.
        x_boundary_inds = np.where(x == rois[:, 2])[0]
        y_boundary_inds = np.where(y == rois[:, 3])[0]
        x = (x - offset_x) * scale_x
        x = np.floor(x)
        if len(x_boundary_inds) > 0:
            x[x_boundary_inds] = cfg.KRCNN.HEATMAP_SIZE - 1

        y = (y - offset_y) * scale_y
        y = np.floor(y)
        if len(y_boundary_inds) > 0:
            y[y_boundary_inds] = cfg.KRCNN.HEATMAP_SIZE - 1

        valid_loc = np.logical_and(
            np.logical_and(x >= 0, y >= 0),
            np.logical_and(
                x < cfg.KRCNN.HEATMAP_SIZE, y < cfg.KRCNN.HEATMAP_SIZE))

        valid = np.logical_and(valid_loc, vis)
        valid = valid.astype(np.int32)

        lin_ind = y * cfg.KRCNN.HEATMAP_SIZE + x
        heatmaps[:, kp] = lin_ind * valid
        weights[:, kp] = valid

    return heatmaps, weights


def scores_to_probs(scores):
    """Transforms CxHxW of scores to probabilities spatially."""
    channels = scores.shape[0]
    for c in range(channels):
        temp = scores[c, :, :]
        max_score = temp.max()
        temp = np.exp(temp - max_score) / np.sum(np.exp(temp - max_score))
        scores[c, :, :] = temp
    return scores


def nms_oks(kp_predictions, rois, thresh):
    """Nms based on kp predictions."""

    scores = np.mean(kp_predictions[:, 2, :], axis=1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovr = compute_oks(
            kp_predictions[i], rois[i], kp_predictions[order[1:]],
            rois[order[1:]])
        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def compute_oks(src_keypoints, src_roi, dst_keypoints, dst_roi):
    """ Compute OKS for predicted keypoints wrt gt_keypoints.
    src_keypoints: 4xK
    src_roi: 4x1
    dst_keypoints: Nx4xK
    dst_roi: Nx4
    """

    sigmas = np.array([
        .26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87,
        .87, .89, .89]) / 10.0
    vars = (sigmas * 2)**2

    # area
    src_area = (src_roi[2] - src_roi[0] + 1) * (src_roi[3] - src_roi[1] + 1)

    # measure the per-keypoint distance if keypoints visible
    dx = dst_keypoints[:, 0, :] - src_keypoints[0, :]
    dy = dst_keypoints[:, 1, :] - src_keypoints[1, :]

    e = (dx**2 + dy**2) / vars / (src_area + np.spacing(1)) / 2
    e = np.sum(np.exp(-e), axis=1) / e.shape[1]

    return e


def compute_head_size(kps, kpt_names):
    """
    Based on
    https://github.com/leonid-pishchulin/poseval/blob/954d8d84f459e942a185f835fc2a0fbdee5ce354/py/eval_helpers.py#L73  # noQA
    """
    head_top = kps[:2, kpt_names.index('head_top')]
    head_bottom = kps[:2, kpt_names.index('head_bottom')]
    # originally was 0.6 x hypotenuse of the head, but don't have those kpts
    return np.linalg.norm(head_top - head_bottom) + 1  # to avoid 0s


def pck_distance(kps_a, kps_b, kpt_names, dist_thresh=0.5):
    """
    Compute distance between the 2 keypoints, where each is represented
    as a 3x17 or 4x17 np.ndarray.
    Ideally use kps_a as GT, if one is GT
    """
    head_size = compute_head_size(kps_a, kpt_names)
    # distance between all points
    # TODO(rgirdhar): Some of these keypoints might be invalid -- not labeled
    # in the dataset. Might need to do something about that...
    normed_dist = np.linalg.norm(kps_a[:2] - kps_b[:2], axis=0) / head_size
    match = normed_dist < dist_thresh
    pck = np.sum(match) / match.size
    pck_dist = 1.0 - pck
    return pck_dist
