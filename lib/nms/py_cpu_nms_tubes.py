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


def py_cpu_nms_tubes(dets, thresh):
    """Pure Python NMS baseline."""
    T = (dets.shape[1] - 1) // 4

    areas = {}
    for t in range(T):
        x1 = dets[:, 4 * t + 0]
        y1 = dets[:, 4 * t + 1]
        x2 = dets[:, 4 * t + 2]
        y2 = dets[:, 4 * t + 3]

        areas[t] = (x2 - x1 + 1) * (y2 - y1 + 1)

    scores = dets[:, -1]
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ovT = 0.0
        for t in range(T):
            xx1 = np.maximum(dets[i, 4 * t + 0], dets[order[1:], 4 * t + 0])
            yy1 = np.maximum(dets[i, 4 * t + 1], dets[order[1:], 4 * t + 1])
            xx2 = np.minimum(dets[i, 4 * t + 2], dets[order[1:], 4 * t + 2])
            yy2 = np.minimum(dets[i, 4 * t + 3], dets[order[1:], 4 * t + 3])

            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[t][i] + areas[t][order[1:]] - inter)
            ovT += ovr
        ovT /= T
        inds = np.where(ovT <= thresh)[0]
        order = order[inds + 1]

    return keep
