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
import pycocotools.mask as mask_util


def flip_segms(segms, width):
    def _flip(poly, width):
        flipped_poly = np.array(poly)
        flipped_poly[0::2] = width - np.array(poly[0::2]) - 1
        return flipped_poly.tolist()
    flipped_segms = []
    for segm in segms:
        if type(segm) == list:
            flipped_segms.append([_flip(poly, width) for poly in segm])
        else:
            # TODO segm encoded as RLE (crowd label).  Ignore it for now
            flipped_segms.append(segm)
    return flipped_segms


def polys_to_mask(polygons, height, width):
    rle = mask_util.frPyObjects(polygons, height, width)
    mask = np.array(mask_util.decode(rle), dtype=np.float32)
    mask = np.sum(mask, axis=2)
    mask = np.array(mask > 0, dtype=np.float32)
    return mask


def mask_to_bbox(mask):
    xs = np.where(np.sum(mask, axis=0) > 0)[0]
    ys = np.where(np.sum(mask, axis=1) > 0)[0]

    if len(xs) == 0 or len(ys) == 0:
        return None

    x0 = xs[0]
    x1 = xs[-1]
    y0 = ys[0]
    y1 = ys[-1]
    return np.array((x0, y0, x1, y1), dtype=np.float32)


def polys_to_mask_wrt_box(polygons, box, M):
    w = box[2] - box[0]
    h = box[3] - box[1]

    w = np.maximum(w, 1)
    h = np.maximum(h, 1)

    polygons_norm = []
    for poly in polygons:
        p = np.array(poly, dtype=np.float32)
        p[0::2] = (p[0::2] - box[0]) * M / w
        p[1::2] = (p[1::2] - box[1]) * M / h
        polygons_norm.append(p)

    rle = mask_util.frPyObjects(polygons_norm, M, M)
    mask = np.array(mask_util.decode(rle), dtype=np.float32)
    mask = np.sum(mask, axis=2)
    mask = np.array(mask > 0, dtype=np.float32)
    return mask


def polys_to_boxes(polys):
    boxes_from_polys = np.zeros((len(polys), 4), dtype=np.float32)
    for i in range(len(polys)):
        poly = polys[i]
        x0 = min(min(p[::2]) for p in poly)
        x1 = max(max(p[::2]) for p in poly)
        y0 = min(min(p[1::2]) for p in poly)
        y1 = max(max(p[1::2]) for p in poly)
        boxes_from_polys[i, :] = [x0, y0, x1, y1]

    return boxes_from_polys
