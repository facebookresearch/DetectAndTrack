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

from core.config import cfg


def add_roi_2mlp_head(model, blob_in, dim_in, spatial_scale):
    # TODO(rbg): this uses hard-coded FAST_RCNN configs, so it cannot be used
    # in different places
    hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in, 'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)
    # TODO(rgirdhar): The temporal dim here might not be same as NUM_FRAMES
    # (eg in case of temporal striding). Need a better way to handle
    time_dim = cfg.VIDEO.NUM_FRAMES_MID if cfg.MODEL.VIDEO_ON and \
               cfg.VIDEO.BODY_HEAD_LINK == '' else 1
    model.FC(roi_feat, 'fc6', time_dim * dim_in * roi_size * roi_size, hidden_dim)
    model.Relu('fc6', 'fc6')
    model.FC('fc6', 'fc7', hidden_dim, hidden_dim)
    model.Relu('fc7', 'fc7')
    return 'fc7', hidden_dim, spatial_scale


def add_roi_2mlp_glu_head(model, blob_in, dim_in, spatial_scale):
    # TODO(rbg): this uses hard-coded FAST_RCNN configs, so it cannot be used
    # in different places
    hidden_dim = cfg.FAST_RCNN.MLP_HEAD_DIM
    roi_size = cfg.FAST_RCNN.ROI_XFORM_RESOLUTION
    roi_feat = model.RoIFeatureTransform(
        blob_in, 'roi_feat',
        blob_rois='rois',
        method=cfg.FAST_RCNN.ROI_XFORM_METHOD,
        resolution=roi_size,
        sampling_ratio=cfg.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO,
        spatial_scale=spatial_scale)

    fc6_pre = model.FC(roi_feat, 'fc6_pre', dim_in * roi_size**2, hidden_dim)
    fc6_act = model.FC(roi_feat, 'fc6_act', dim_in * roi_size**2, hidden_dim)
    fc6_gate = model.net.Sigmoid(fc6_act, fc6_act)
    fc6 = model.net.Mul([fc6_pre, fc6_gate], 'fc6')

    fc7_pre = model.FC(fc6, 'fc7_pre', hidden_dim, hidden_dim)
    fc7_act = model.FC(fc6, 'fc7_act', hidden_dim, hidden_dim)
    fc7_gate = model.net.Sigmoid(fc7_act, fc7_act)
    model.net.Mul([fc7_pre, fc7_gate], 'fc7')

    return 'fc7', hidden_dim, spatial_scale
