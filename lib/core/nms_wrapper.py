##############################################################
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################

# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

# from nms.cpu_nms import cpu_nms, cpu_soft_nms
# from new Detectron
from utils.cython_nms import nms as cpu_nms, soft_nms as cpu_soft_nms
from nms.py_cpu_nms_tubes import py_cpu_nms_tubes


def soft_nms(
    dets, sigma=0.5, overlap_thresh=0.3, score_thresh=0.001, method='linear'
):
    if dets.shape[0] == 0:
        return dets
    if dets.shape[1] > 5:
        raise NotImplementedError('Need to handle tubes..')

    methods = {'hard': 0, 'linear': 1, 'gaussian': 2}
    assert method in methods, 'Unknown soft_nms method: {}'.format(method)

    dets = cpu_soft_nms(
        np.ascontiguousarray(dets, dtype=np.float32),
        np.float32(sigma),
        np.float32(overlap_thresh),
        np.float32(score_thresh),
        np.uint8(methods[method]))
    return dets


def nms(dets, thresh, soft_nms=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    if dets.shape[1] > 5:  # normal box is (x, y, x, y, score), so 5-dim
        return tube_nms(dets, thresh)
    else:
        return cpu_nms(dets, thresh)


def tube_nms(dets, thresh):
    """ Perform NMS over tubes (in videos).
        Args:
            dets (np.ndarray of shape N x (K * 4 + 1)): K is the number of
            frames, so all boxes of that frame are concatenated one after other
            in the horizonal dimension. +1 is for the score.
            Rest arguments same as nms.
        Returns:
            Same as nms ("keep" indexes).
    """
    return py_cpu_nms_tubes(dets, thresh)
