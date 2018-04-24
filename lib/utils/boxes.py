##############################################################
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################

# --------------------------------------------------------
# Fast/er R-CNN
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from core.config import cfg
from utils.cython_bbox import bbox_overlaps as cython_bbox_overlaps


def split_tube_into_boxes(tube, T=None):
    """ N tubes are represented as a (N, 4 * T,) dimensional matrix,
    or sometimes (N, 4 * T * num_classes) matrix. In the latter case, T should
    be passed as input. In some cases, there can be a score value at the end of
    the tube vector, which will be copied to each split box. """
    N = tube.shape[0]
    if tube.shape[1] % 4 == 0:
        scores = np.zeros((N, 0))
    elif (tube.shape[1] - 1) % 4 == 0:
        scores = tube[:, (-1,)]
        tube = tube[:, :-1]
    else:
        raise ValueError('Invalid tube dimensions {}'.format(tube.shape))
    T = T or tube.shape[-1] // 4
    boxes = []
    if 4 * T != tube.shape[-1]:
        # Means it is the box-per-class kinda representation
        # Take ith box from each class
        assert(tube.shape[-1] % (4 * T) == 0)
        num_classes = tube.shape[-1] // (4 * T)
        for t in range(T):
            box_rep = np.zeros((tube.shape[0], 4 * num_classes))
            for cid in range(num_classes):
                box_rep[:, cid * 4: (cid + 1) * 4] = \
                    tube[:, cid * 4 * T: (cid + 1) * 4 * T][:, t * 4: (t + 1) * 4]
            boxes.append(box_rep)
    else:
        for t in range(T):
            boxes.append(tube[..., t * 4: (t + 1) * 4])
    # Add the scores back, if there were any
    boxes = [np.hstack((box, scores)) for box in boxes]
    return boxes, T


def bbox_overlaps(boxes, query_boxes):
    """ Use the cython overlap implementation to compute the overlap,
    and average the overlaps over time for tubes. """
    parts, _ = split_tube_into_boxes(boxes)
    query_parts, _ = split_tube_into_boxes(query_boxes)
    return np.mean(np.stack([
        cython_bbox_overlaps(
            part.astype(np.float32, copy=False),
            query_part.astype(np.float32, copy=False))
        for part, query_part in zip(parts, query_parts)]), axis=0)


def boxes_area(boxes):
    # ::4 to allow for tubes. Averages each corr. box area over the whole tube
    w = (boxes[:, 2::4] - boxes[:, 0::4] + 1)
    h = (boxes[:, 3::4] - boxes[:, 1::4] + 1)
    areas = np.mean(w * h, axis=1)
    assert np.all(areas >= 0), 'Negative areas founds'
    return areas


def unique_boxes(boxes, scale=1.0):
    """Return indices of unique boxes."""
    assert boxes.shape[1] == 4, 'Func doesnot support tubes yet'
    v = np.array([1, 1e3, 1e6, 1e9])
    hashes = np.round(boxes * scale).dot(v)
    _, index = np.unique(hashes, return_index=True)
    return np.sort(index)


def xywh_to_xyxy(boxes):
    """Convert [x y w h] box format to [x1 y1 x2 y2] format."""
    assert boxes.shape[1] == 4, 'Func doesnot support tubes yet'
    return np.hstack((boxes[:, 0:2], boxes[:, 0:2] + boxes[:, 2:4] - 1))


def xyxy_to_xywh(boxes):
    """Convert [x1 y1 x2 y2] box format to [x y w h] format."""
    assert boxes.shape[1] == 4, 'Func doesnot support tubes yet'
    return np.hstack((boxes[:, 0:2], boxes[:, 2:4] - boxes[:, 0:2] + 1))


def validate_boxes(boxes, width=0, height=0):
    """Check that a set of boxes are valid."""
    assert boxes.shape[1] == 4, 'Func doesnot support tubes yet'
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    return (
        (x1 >= 0).all() and
        (y1 >= 0).all() and
        (x2 >= x1).all() and
        (y2 >= y1).all() and
        (x2 < width).all() and
        (y2 < height).all())


def filter_small_boxes(boxes, min_size):
    assert boxes.shape[1] == 4, 'Func doesnot support tubes yet'
    w = boxes[:, 2] - boxes[:, 0]
    h = boxes[:, 3] - boxes[:, 1]
    keep = np.where((w >= min_size) & (h > min_size))[0]
    return keep


def clip_boxes_to_image(boxes, height, width):
    assert boxes.shape[1] == 4, 'Func doesnot support tubes yet'
    boxes[:, [0, 2]] = np.minimum(width - 1., np.maximum(0., boxes[:, [0, 2]]))
    boxes[:, [1, 3]] = np.minimum(height - 1., np.maximum(0., boxes[:, [1, 3]]))
    return boxes


def clip_xyxy_to_image(x1, y1, x2, y2, height, width):
    x1 = np.minimum(width - 1., np.maximum(0., x1))
    y1 = np.minimum(height - 1., np.maximum(0., y1))
    x2 = np.minimum(width - 1., np.maximum(0., x2))
    y2 = np.minimum(height - 1., np.maximum(0., y2))
    return x1, y1, x2, y2


def bbox_transform(boxes, deltas, weights):
    """Forward transform that maps proposal boxes to ground-truth boxes using
    bounding-box regression deltas. See bbox_transform_inv for a description of
    the weights argument.
    """
    if boxes.shape[1] > 4:
        return tube_transform(boxes, deltas, weights)
    if boxes.shape[0] == 0:
        return np.zeros((0, deltas.shape[1]), dtype=deltas.dtype)

    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    wx, wy, ww, wh = weights
    dx = deltas[:, 0::4] / wx
    dy = deltas[:, 1::4] / wy
    dw = deltas[:, 2::4] / ww
    dh = deltas[:, 3::4] / wh

    # Prevent sending too large values into np.exp()
    dw = np.minimum(dw, cfg.BBOX_XFORM_CLIP)
    dh = np.minimum(dh, cfg.BBOX_XFORM_CLIP)

    pred_ctr_x = dx * widths[:, np.newaxis] + ctr_x[:, np.newaxis]
    pred_ctr_y = dy * heights[:, np.newaxis] + ctr_y[:, np.newaxis]
    pred_w = np.exp(dw) * widths[:, np.newaxis]
    pred_h = np.exp(dh) * heights[:, np.newaxis]

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0::4] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1::4] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2::4] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3::4] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes


def tube_transform(boxes, deltas, weights):
    """ Use the bbox_transform to compute the transform,
    and apply to all the boxes. """
    boxes_parts, time_dim = split_tube_into_boxes(boxes)
    deltas_parts, _ = split_tube_into_boxes(deltas, time_dim)
    all_tx = [bbox_transform(boxes_part, deltas_part, weights) for
              boxes_part, deltas_part in zip(boxes_parts, deltas_parts)]
    # Now need to interleave the boxes in the same way they were un-interleaved
    # in the split_tube_into_boxes function
    nclasses = all_tx[0].shape[-1] // 4
    time_dim = len(all_tx)
    res = np.zeros(deltas.shape, dtype=deltas.dtype)
    for cid in range(nclasses):
        for t in range(time_dim):
            res[:, cid * 4 * time_dim: (cid + 1) * 4 * time_dim][
                :, t * 4: (t + 1) * 4] = all_tx[t][:, cid * 4: (cid + 1) * 4]
    return res


def bbox_transform_inv(ex_rois, gt_rois, weights):
    """Backward transform that computes bounding-box regression deltas given
    proposal boxes and ground-truth boxes. The weights argument should be a
    4-tuple of multiplicative weights that are applied to the regression target.
    """
    if ex_rois.shape[1] > 4:
        return tube_transform_inv(ex_rois, gt_rois, weights)
    ex_widths = ex_rois[:, 2] - ex_rois[:, 0] + 1.0
    ex_heights = ex_rois[:, 3] - ex_rois[:, 1] + 1.0
    ex_ctr_x = ex_rois[:, 0] + 0.5 * ex_widths
    ex_ctr_y = ex_rois[:, 1] + 0.5 * ex_heights

    gt_widths = gt_rois[:, 2] - gt_rois[:, 0] + 1.0
    gt_heights = gt_rois[:, 3] - gt_rois[:, 1] + 1.0
    gt_ctr_x = gt_rois[:, 0] + 0.5 * gt_widths
    gt_ctr_y = gt_rois[:, 1] + 0.5 * gt_heights

    wx, wy, ww, wh = weights
    targets_dx = wx * (gt_ctr_x - ex_ctr_x) / ex_widths
    targets_dy = wy * (gt_ctr_y - ex_ctr_y) / ex_heights
    targets_dw = ww * np.log(gt_widths / ex_widths)
    targets_dh = wh * np.log(gt_heights / ex_heights)

    targets = np.vstack(
        (targets_dx, targets_dy, targets_dw, targets_dh)).transpose()
    return targets


def tube_transform_inv(ex_rois, gt_rois, weights):
    """ Tube extension of the box rois. """
    assert(ex_rois.shape[1] == gt_rois.shape[1])
    ex_rois_parts, _ = split_tube_into_boxes(ex_rois)
    gt_rois_parts, _ = split_tube_into_boxes(gt_rois)
    all_tx = [bbox_transform_inv(ex_rois_part, gt_rois_part, weights) for
              ex_rois_part, gt_rois_part in zip(ex_rois_parts, gt_rois_parts)]
    return np.concatenate(all_tx, axis=1)


def clip_tiled_boxes(boxes, im_shape):
    """Clip boxes to image boundaries. im_shape is [height, width]."""
    # x1 >= 0
    boxes[:, 0::4] = np.maximum(np.minimum(boxes[:, 0::4], im_shape[1] - 1), 0)
    # y1 >= 0
    boxes[:, 1::4] = np.maximum(np.minimum(boxes[:, 1::4], im_shape[0] - 1), 0)
    # x2 < im_shape[1]
    boxes[:, 2::4] = np.maximum(np.minimum(boxes[:, 2::4], im_shape[1] - 1), 0)
    # y2 < im_shape[0]
    boxes[:, 3::4] = np.maximum(np.minimum(boxes[:, 3::4], im_shape[0] - 1), 0)
    return boxes


def expand_boxes(boxes, ratio):
    assert boxes.shape[1] == 4, 'Func doesnot support tubes yet'
    w_half = (boxes[:, 2] - boxes[:, 0]) * .5
    h_half = (boxes[:, 3] - boxes[:, 1]) * .5
    x_c = (boxes[:, 2] + boxes[:, 0]) * .5
    y_c = (boxes[:, 3] + boxes[:, 1]) * .5

    w_half *= ratio
    h_half *= ratio

    boxes_exp = np.zeros(boxes.shape)
    boxes_exp[:, 0] = x_c - w_half
    boxes_exp[:, 2] = x_c + w_half
    boxes_exp[:, 1] = y_c - h_half
    boxes_exp[:, 3] = y_c + h_half

    return boxes_exp


def flip_boxes(boxes, im_width):
    """Flips boxes horizontally."""
    assert boxes.shape[1] == 4, 'Func doesnot support tubes yet'
    boxes_flipped = boxes.copy()
    boxes_flipped[:, 0::4] = im_width - boxes[:, 2::4] - 1
    boxes_flipped[:, 2::4] = im_width - boxes[:, 0::4] - 1
    assert np.all(boxes_flipped[:, 2::4] >= boxes_flipped[:, 0::4])
    return boxes_flipped


def aspect_ratio(boxes, aspect_ratio):
    """Performs width-relative aspect ratio transformation."""
    assert boxes.shape[1] == 4, 'Func doesnot support tubes yet'
    boxes_ar = boxes.copy()
    boxes_ar[:, 0::4] = aspect_ratio * boxes[:, 0::4]
    boxes_ar[:, 2::4] = aspect_ratio * boxes[:, 2::4]
    return boxes_ar


def box_voting(top_dets, all_dets, thresh):
    """Apply bounding-box voting to refine `top_dets` by voting with `all_dets`.
    See: https://arxiv.org/abs/1505.01749.
    """
    # top_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    # all_dets is [N, 5] each row is [x1 y1 x2 y2, sore]
    top_dets_out = top_dets.copy()
    top_boxes = top_dets[:, :4]
    all_boxes = all_dets[:, :4]
    all_scores = all_dets[:, 4]
    top_to_all_overlaps = bbox_overlaps(top_boxes, all_boxes)
    for k in range(top_dets_out.shape[0]):
        inds_to_vote = np.where(top_to_all_overlaps[k] >= thresh)[0]
        boxes_to_vote = all_boxes[inds_to_vote, :]
        ws = all_scores[inds_to_vote]
        top_dets_out[k, :4] = np.average(boxes_to_vote, axis=0, weights=ws)
    return top_dets_out
