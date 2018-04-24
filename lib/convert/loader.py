##############################################################
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################

from __future__ import absolute_import, division, print_function, unicode_literals

import scipy.io as sio
import collections
import numpy as np


VideoAnnot = collections.namedtuple(
    'VideoAnnot', [
        'im_name',  # Path to the image
        'is_labeled',  # True if labeled
        'boxes',  # all the boxes and poses in this image (if labeled)
    ])
VideoAnnotBox = collections.namedtuple(
    'VideoAnnotBox', [
        'head',  # (x1, y1, x2, y2) 4-tuple
        'track_id',  # integer
        'pose',  # np.array of 17x3 dimension
    ])


def load_mat(fpath):
    """
    Args:
      fpath (str): String path to the released MAT annotation file
    Returns:
      Structure with the data in the MAT file
    """
    res_annots = []
    data = sio.loadmat(fpath, squeeze_me=True, struct_as_record=False)
    annolist = data['annolist']
    if not hasattr(annolist, '__iter__'):  # is not iterable
        if isinstance(annolist, sio.matlab.mio5_params.mat_struct):
            annolist = [annolist]
        else:
            print('Unable to read annolist from {}'.format(fpath))
            annolist = []
    for ann_id in range(len(annolist)):
        annot = annolist[ann_id]
        res_annot_im_name = annot.image.name
        res_annot_is_labeled = (int(annot.is_labeled) == 1)
        res_annot_boxes = []
        if res_annot_is_labeled:
            all_boxes = annot.annorect
            if isinstance(all_boxes, np.ndarray):
                all_boxes = all_boxes.tolist()
            if isinstance(all_boxes, sio.matlab.mio5_params.mat_struct):
                all_boxes = [all_boxes]
            nboxes = len(all_boxes)
            for box_id in range(nboxes):
                res_annot_box = VideoAnnotBox([], -1, [])
                head = (all_boxes[box_id].x1,
                        all_boxes[box_id].y1,
                        all_boxes[box_id].x2,
                        all_boxes[box_id].y2)
                track_id = all_boxes[box_id].track_id
                pose = np.zeros((15, 3))
                try:
                    points = all_boxes[box_id].annopoints.point
                except:
                    points = []
                if isinstance(points, np.ndarray):
                    points = points.tolist()
                if isinstance(points, sio.matlab.mio5_params.mat_struct):
                    points = [points]
                for point in points:
                    ptid = point.id
                    ptx = point.x
                    pty = point.y
                    pose[ptid, ...] = (ptx, pty, 2)
                res_annot_box = VideoAnnotBox(head, track_id, pose)
                res_annot_boxes.append(res_annot_box)
        res_annots.append(VideoAnnot(
            res_annot_im_name, res_annot_is_labeled, res_annot_boxes))
    return res_annots
