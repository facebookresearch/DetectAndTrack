##############################################################
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################

import numpy as np
import itertools

from core.config import cfg

# Verify that we compute the same anchors as Shaoqing's matlab implementation:
#
#    >> load output/rpn_cachedir/faster_rcnn_VOC2007_ZF_stage1_rpn/anchors.mat
#    >> anchors
#
#    anchors =
#
#       -83   -39   100    56
#      -175   -87   192   104
#      -359  -183   376   200
#       -55   -55    72    72
#      -119  -119   136   136
#      -247  -247   264   264
#       -35   -79    52    96
#       -79  -167    96   184
#      -167  -343   184   360

#array([[ -83.,  -39.,  100.,   56.],
#       [-175.,  -87.,  192.,  104.],
#       [-359., -183.,  376.,  200.],
#       [ -55.,  -55.,   72.,   72.],
#       [-119., -119.,  136.,  136.],
#       [-247., -247.,  264.,  264.],
#       [ -35.,  -79.,   52.,   96.],
#       [ -79., -167.,   96.,  184.],
#       [-167., -343.,  184.,  360.]])


def generate_anchors(
        stride=16, sizes=(32, 64, 128, 256, 512), aspect_ratios=(0.5, 1, 2),
        time_dim=1):
    """Generates a matrix of anchor boxes in (x1, y1, x2, y2) format. Anchors
    are centered on stride / 2, have (approximate) sqrt areas of the specified
    sizes, and aspect ratios as given.
    time_dim (added by rgirdhar): defines number of time-frames the anchor
    is for .. For now just repmat the anchors that many times.
    """
    return _generate_anchors(
        stride, np.array(sizes, dtype=np.float) / stride,
        np.array(aspect_ratios, dtype=np.float), time_dim)


def _generate_anchors(base_size, scales, aspect_ratios, time_dim):
    """Generate anchor (reference) windows by enumerating aspect ratios X
    scales wrt a reference (0, 0, base_size - 1, base_size - 1) window.
    """
    anchor = np.array([1, 1, base_size, base_size], dtype=np.float) - 1
    anchors = _ratio_enum(anchor, aspect_ratios)
    anchors = np.vstack([_scale_enum(anchors[i, :], scales)
                         for i in range(anchors.shape[0])])
    # time_dim defines the number of time-frames I need the tube to be for.
    # By default, single image case is time_dim = 1
    if cfg.VIDEO.RPN_TUBE_GEN_STYLE == 'replicate':
        anchors = np.tile(anchors, [1, time_dim])
    elif cfg.VIDEO.RPN_TUBE_GEN_STYLE == 'combinations':
        anchors = np.array([
            sum(item, []) for item in itertools.combinations_with_replacement(
                anchors.tolist(), time_dim)])
    elif cfg.VIDEO.RPN_TUBE_GEN_STYLE == 'permutations':
        anchors = np.array([
            sum(item, []) for item in itertools.permutations(
                anchors.tolist(), time_dim)])
    else:
        raise NotImplementedError('Unknown {}'.format(cfg.VIDEO.RPN_TUBE_GEN_STYLE))
    return anchors


def _whctrs(anchor):
    """
    Return width, height, x center, and y center for an anchor (window).
    """

    w = anchor[2] - anchor[0] + 1
    h = anchor[3] - anchor[1] + 1
    x_ctr = anchor[0] + 0.5 * (w - 1)
    y_ctr = anchor[1] + 0.5 * (h - 1)
    return w, h, x_ctr, y_ctr


def _mkanchors(ws, hs, x_ctr, y_ctr):
    """
    Given a vector of widths (ws) and heights (hs) around a center
    (x_ctr, y_ctr), output a set of anchors (windows).
    """

    ws = ws[:, np.newaxis]
    hs = hs[:, np.newaxis]
    anchors = np.hstack((x_ctr - 0.5 * (ws - 1),
                         y_ctr - 0.5 * (hs - 1),
                         x_ctr + 0.5 * (ws - 1),
                         y_ctr + 0.5 * (hs - 1)))
    return anchors


def _ratio_enum(anchor, ratios):
    """
    Enumerate a set of anchors for each aspect ratio wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    size = w * h
    size_ratios = size / ratios
    ws = np.round(np.sqrt(size_ratios))
    hs = np.round(ws * ratios)
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def _scale_enum(anchor, scales):
    """
    Enumerate a set of anchors for each scale wrt an anchor.
    """

    w, h, x_ctr, y_ctr = _whctrs(anchor)
    ws = w * scales
    hs = h * scales
    anchors = _mkanchors(ws, hs, x_ctr, y_ctr)
    return anchors


def time_extend_shifts(shifts, time_dim):
    """
    Extend the shifts generated for shifting anchors, for time_dim time
    intervals.
    """
    shifts = np.tile(shifts, [1, time_dim])
    return shifts
