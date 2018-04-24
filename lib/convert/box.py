##############################################################
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np


def expand_boxes(boxes, ratio):
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


def compute_boxes_from_pose(poses):
    """
    Args:
        poses (list of list of list of floats):
            list of poses in each frame, each list contains list of poses in
            that frame, where each pose is a 17*3 element list (COCO style).
    Returns:
        boxes: (list of list of list of floats):
            list of boxes in each frame, each list contains a list of boxes in
            that frame, where each pose is [x, y, w, h] list.
    Added by rgirdhar
    """
    boxes = []
    for frame_poses in poses:
        if len(frame_poses) == 0:
            boxes.append([])
            continue
        frame_boxes = []
        frame_poses_np = np.array(frame_poses)
        frame_poses_np = frame_poses_np.reshape((-1, 17, 3))
        # only consider the points that are marked "2", i.e. labeled and visible
        valid_pts = frame_poses_np[:, :, 2] == 2
        for pose_id in range(frame_poses_np.shape[0]):
            valid_pose = frame_poses_np[pose_id, valid_pts[pose_id], :]
            # TODO(rgirdhar): Need to figure what to do here... Maybe just
            # use the head box heuristic or something to proxy the box..
            # For now just letting it get a random box
            if valid_pose.shape[0] == 0:
                frame_boxes.append([0, 0, 0, 0])
                continue
            # gen a xmin, ymin, xmax, ymax
            box = np.array([
                np.min(valid_pose[:, 0]),
                np.min(valid_pose[:, 1]),
                # The +1 ensures the box is at least 1x1 in size. Such
                # small boxes will be later removed anyway I think
                np.max(valid_pose[:, 0]) + 1,
                np.max(valid_pose[:, 1]) + 1,
            ])
            # Expand by 20%
            box = expand_boxes(np.expand_dims(box, 0), 1.2)[0]
            # conver to x,y,w,h; same as COCO json format (which is what it is
            # in, at this point)
            frame_boxes.append([
                box[0], box[1], box[2] - box[0], box[3] - box[1]])
        boxes.append(frame_boxes)
    return boxes
