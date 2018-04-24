##############################################################
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################

def get_posetrack_kpt_ordering():
    ordering = [
        'right_ankle',
        'right_knee',
        'right_hip',
        'left_hip',
        'left_knee',
        'left_ankle',
        'right_wrist',
        'right_elbow',
        'right_shoulder',
        'left_shoulder',
        'left_elbow',
        'left_wrist',
        'head_bottom',
        'nose',
        'head_top',
    ]
    joints = [
        ('head_top', 'nose'),
        ('nose', 'head_bottom'),
        ('head_bottom', 'left_shoulder'),
        ('head_bottom', 'right_shoulder'),
        ('left_shoulder', 'left_elbow'),
        ('left_elbow', 'left_wrist'),
        ('right_elbow', 'right_wrist'),
        ('right_shoulder', 'right_elbow'),
        ('head_bottom', 'left_hip'),
        ('head_bottom', 'right_hip'),
        ('left_hip', 'left_knee'),
        ('left_knee', 'left_ankle'),
        ('right_hip', 'right_knee'),
        ('right_knee', 'right_ankle'),
    ]
    return ordering, joints
