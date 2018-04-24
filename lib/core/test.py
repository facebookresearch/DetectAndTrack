##############################################################
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################

"""Test a Fast R-CNN network on an image."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2
from collections import defaultdict

from caffe2.python import core, workspace
import pycocotools.mask as mask_util

from core.config import cfg
import utils.boxes as box_utils
import utils.image as image_utils
import utils.keypoints as keypoint_utils
from utils.timer import Timer
from core.nms_wrapper import nms, soft_nms
import utils.blob as blob_utils
import modeling.FPN as fpn

import logging
logger = logging.getLogger(__name__)

# OpenCL is enabled by default in OpenCV3 and it is not thread-safe leading
# to huge GPU memory allocations.
try:
    cv2.ocl.setUseOpenCL(False)
except AttributeError:
    pass


def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (list of ndarray): a list of color images in BGR order. In case of
        video it is a list of frames, else is is a list with len = 1.

    Returns:
        blob (ndarray): a data blob holding an image pyramid (or video pyramid)
        im_scale_factors (ndarray): array of image scales (relative to im) used
            in the image pyramid
    """
    all_processed_ims = []  # contains a a list for each frame, for each scale
    all_im_scale_factors = []
    for frame in im:
        processed_ims, im_scale_factors = blob_utils.prep_im_for_blob(
            frame, cfg.PIXEL_MEANS, cfg.TEST.SCALES, cfg.TEST.MAX_SIZE)
        all_processed_ims.append(processed_ims)
        all_im_scale_factors.append(im_scale_factors)
    # All the im_scale_factors will be the same, so just take the first one
    for el in all_im_scale_factors:
        assert(all_im_scale_factors[0] == el)
    im_scale_factors = all_im_scale_factors[0]
    # Now get all frames with corresponding scale next to each other
    processed_ims = []
    for i in range(len(all_processed_ims[0])):
        for frames_at_specific_scale in all_processed_ims:
            processed_ims.append(frames_at_specific_scale[i])
    # Now processed_ims contains
    # [frame1_scale1, frame2_scale1..., frame1_scale2, frame2_scale2...] etc
    blob = blob_utils.im_list_to_blob(processed_ims)
    return blob, np.array(im_scale_factors)


def _get_rois_blob(im_rois, im_scale_factors):
    """Converts RoIs into network inputs.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        im_scale_factors (list): scale factors as returned by _get_image_blob

    Returns:
        blob (ndarray): R x 5 matrix of RoIs in the image pyramid with columns
            [level, x1, y1, x2, y2]
    """
    rois, levels = _project_im_rois(im_rois, im_scale_factors)
    rois_blob = np.hstack((levels, rois))
    return rois_blob.astype(np.float32, copy=False)


def _project_im_rois(im_rois, scales):
    """Project image RoIs into the image pyramid built by _get_image_blob.

    Arguments:
        im_rois (ndarray): R x 4 matrix of RoIs in original image coordinates
        scales (list): scale factors as returned by _get_image_blob

    Returns:
        rois (ndarray): R x 4 matrix of projected RoI coordinates
        levels (ndarray): image pyramid levels used by each projected RoI
    """
    im_rois = im_rois.astype(np.float, copy=False)

    if len(scales) > 1:
        # Works for tubes as well, as it uses the first box's area -- which is
        # a reasonable approx for the tube area
        widths = im_rois[:, 2] - im_rois[:, 0] + 1
        heights = im_rois[:, 3] - im_rois[:, 1] + 1

        areas = widths * heights
        scaled_areas = areas[:, np.newaxis] * (scales[np.newaxis, :] ** 2)
        diff_areas = np.abs(scaled_areas - 224 * 224)
        levels = diff_areas.argmin(axis=1)[:, np.newaxis]
    else:
        levels = np.zeros((im_rois.shape[0], 1), dtype=np.int)

    rois = im_rois * scales[levels]

    return rois, levels


def _add_multilevel_rois_for_test(blobs, name):
    """Distributes a set of RoIs across FPN pyramid levels by creating new level
    specific RoI blobs.

    Arguments:
        blobs (dict): dictionary of blobs
        name (str): a key in 'blobs' identifying the source RoI blob

    Returns:
        [by ref] blobs (dict): new keys named by `name + 'fpn' + level`
            are added to dict each with a value that's an R_level x 5 ndarray of
            RoIs (see _get_rois_blob for format)
    """
    lvl_min = cfg.FPN.ROI_MIN_LEVEL
    lvl_max = cfg.FPN.ROI_MAX_LEVEL
    lvls = fpn.map_rois_to_fpn_levels(blobs[name][:, 1:], lvl_min, lvl_max)
    fpn.add_multilevel_roi_blobs(
        blobs, name, blobs[name], lvls, lvl_min, lvl_max)


def _get_blobs(im, rois):
    """Convert an image and RoIs within that image into network inputs."""
    blobs = {}
    blobs['data'], im_scale_factors = _get_image_blob(im)
    if cfg.MODEL.FASTER_RCNN and rois is None:
        blobs['im_info'] = np.array(
            [[blobs['data'].shape[-2], blobs['data'].shape[-1],
              im_scale_factors[0]]],
            dtype=np.float32)
    if rois is not None:
        blobs['rois'] = _get_rois_blob(rois, im_scale_factors)
    return blobs, im_scale_factors


def im_detect_bbox(model, im, boxes=None):
    """Bounding box object detection for an image with given box proposals.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im (ndarray): color image to test (in BGR order)
        boxes (ndarray): R x 4 array of object proposals in 0-indexed
            [x1, y1, x2, y2] format, or None if using RPN

    Returns:
        scores (ndarray): R x K array of object class scores for K classes
            (K includes background as object category 0)
        boxes (ndarray): R x 4*K array of predicted bounding boxes
        im_scales (list): list of image scales used in the input blob (as
            returned by _get_blobs and for use with im_detect_mask, etc.)
    """
    inputs, im_scales = _get_blobs(im, boxes)

    # When mapping from image ROIs to feature map ROIs, there's some aliasing
    # (some distinct image ROIs get mapped to the same feature ROI).
    # Here, we identify duplicate feature ROIs, so we only compute features
    # on the unique subset.
    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        raise NotImplementedError('Can not handle tubes, need to extend dedup')
        v = np.array([1, 1e3, 1e6, 1e9, 1e12])
        hashes = np.round(inputs['rois'] * cfg.DEDUP_BOXES).dot(v)
        _, index, inv_index = np.unique(
            hashes, return_index=True, return_inverse=True)
        inputs['rois'] = inputs['rois'][index, :]
        boxes = boxes[index, :]

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS and not cfg.MODEL.FASTER_RCNN:
        _add_multilevel_rois_for_test(inputs, 'rois')

    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v)
    workspace.RunNet(model.net.Proto().name)

    # dump workspace blobs (debugging)
    # if 0:
    #    from utils.io import robust_pickle_dump
    #    import os, sys
    #    saved_blobs = {}
    #    ws_blobs = workspace.Blobs()
    #    for dst_name in ws_blobs:
    #        ws_blob = workspace.FetchBlob(dst_name)
    #        saved_blobs[dst_name] = ws_blob
    #    det_file = os.path.join('/tmp/output/data_dump_inflT1.pkl')
    #    robust_pickle_dump(saved_blobs, det_file)
    #    logger.info("DUMPED BLOBS")
    #    sys.exit(0)

    # Read out blobs
    if cfg.MODEL.FASTER_RCNN:
        assert len(im_scales) == 1, \
            'Only single-image / single-scale batch implemented'
        rois = workspace.FetchBlob(core.ScopedName('rois'))
        # unscale back to raw image space
        boxes = rois[:, 1:] / im_scales[0]

    if cfg.TEST.SVM:
        # use the raw scores before softmax under the assumption they were
        # trained as linear SVMs
        scores = workspace.FetchBlob(core.ScopedName('cls_score')).squeeze()
    else:
        # use softmax estimated probabilities
        scores = workspace.FetchBlob(core.ScopedName('cls_prob')).squeeze()
    # In case there is 1 proposal
    scores = scores.reshape([-1, scores.shape[-1]])
    time_dim = boxes.shape[-1] // 4

    if cfg.TEST.BBOX_REG:
        # Apply bounding-box regression deltas
        box_deltas = workspace.FetchBlob(core.ScopedName('bbox_pred')).squeeze()
        # In case there is 1 proposal
        box_deltas = box_deltas.reshape([-1, box_deltas.shape[-1]])
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            # Remove predictions for bg class (compat with MSRA code)
            box_deltas = box_deltas[:, -4 * time_dim:]
        pred_boxes = box_utils.bbox_transform(
            boxes, box_deltas, cfg.MODEL.BBOX_REG_WEIGHTS)
        pred_boxes = box_utils.clip_tiled_boxes(pred_boxes, im[0].shape)
        if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG:
            pred_boxes = np.tile(pred_boxes, (1, scores.shape[1]))
    else:
        # Simply repeat the boxes, once for each class
        pred_boxes = np.tile(boxes, (1, scores.shape[1]))

    if cfg.DEDUP_BOXES > 0 and not cfg.MODEL.FASTER_RCNN:
        # Map scores and predictions back to the original set of boxes
        scores = scores[inv_index, :]
        pred_boxes = pred_boxes[inv_index, :]

    return scores, pred_boxes, im_scales


def im_detect_bbox_hflip(model, im, box_proposals=None):
    """Performs bbox detection on the horizontally flipped image.
    Function signature is the same as for im_detect_bbox.
    """
    # Compute predictions on the flipped image
    # im is a list now, to be compat with video case
    im_hf = [e[:, ::-1, :] for e in im]
    # Since all frames would be same shape, just take values from 1st
    im_width = im[0].shape[1]

    if not cfg.MODEL.FASTER_RCNN:
        box_proposals_hf = box_utils.flip_boxes(box_proposals, im_width)
    else:
        box_proposals_hf = None

    scores_hf, boxes_hf, im_scales = im_detect_bbox(
        model, im_hf, box_proposals_hf)

    # Invert the detections computed on the flipped image
    boxes_inv = box_utils.flip_boxes(boxes_hf, im_width)

    return scores_hf, boxes_inv, im_scales


def im_detect_bbox_scale(
        model, im, scale, max_size, box_proposals=None, hflip=False):
    """Computes bbox detections at the given scale.
    Returns predictions in the original image space.
    """
    # Remember the original scale
    orig_scales = cfg.TEST.SCALES
    orig_max_size = cfg.TEST.MAX_SIZE

    # Perform detection at the given scale
    cfg.TEST.SCALES = (scale, )
    cfg.TEST.MAX_SIZE = max_size

    if hflip:
        scores_scl, boxes_scl, _ = im_detect_bbox_hflip(
            model, im, box_proposals)
    else:
        scores_scl, boxes_scl, _ = im_detect_bbox(
            model, im, box_proposals)

    # Restore the original scale
    cfg.TEST.SCALES = orig_scales
    cfg.TEST.MAX_SIZE = orig_max_size

    return scores_scl, boxes_scl


def im_detect_bbox_aspect_ratio(
        model, im, aspect_ratio, box_proposals=None, hflip=False):
    """Computes bbox detections at the given width-relative aspect ratio.
    Returns predictions in the original image space.
    """
    # Compute predictions on the transformed image
    im_ar = [image_utils.aspect_ratio_rel(el, aspect_ratio) for el in im]

    if not cfg.MODEL.FASTER_RCNN:
        box_proposals_ar = box_utils.aspect_ratio(box_proposals, aspect_ratio)
    else:
        box_proposals_ar = None

    if hflip:
        scores_ar, boxes_ar, _ = im_detect_bbox_hflip(
            model, im_ar, box_proposals_ar)
    else:
        scores_ar, boxes_ar, _ = im_detect_bbox(
            model, im_ar, box_proposals_ar)

    # Invert the detected boxes
    boxes_inv = box_utils.aspect_ratio(boxes_ar, 1.0 / aspect_ratio)

    return scores_ar, boxes_inv


def im_detect_bbox_aug(model, im, box_proposals=None):
    """Performs bbox detection with test-time augmentations.
    Function signature is the same as for im_detect_bbox.
    """
    assert not cfg.TEST.BBOX_AUG.SCALE_SIZE_DEP, \
        'Size dependent scaling not implemented'
    assert not cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION' or \
        cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION', \
        'Coord heuristic must be union whenever score heuristic is union'
    assert not cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION' or \
        cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION', \
        'Score heuristic must be union whenever coord heuristic is union'
    assert not cfg.MODEL.FASTER_RCNN or \
        cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION', \
        'Union heuristic must be used to combine Faster RCNN predictions'

    # Collect detections computed under different transformations
    scores_ts = []
    boxes_ts = []

    def add_preds_t(scores_t, boxes_t):
        scores_ts.append(scores_t)
        boxes_ts.append(boxes_t)

    # Perform detection on the horizontally flipped image
    if cfg.TEST.BBOX_AUG.H_FLIP:
        scores_hf, boxes_hf, _im_scales_hf = im_detect_bbox_hflip(
            model, im, box_proposals)
        add_preds_t(scores_hf, boxes_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.BBOX_AUG.SCALES:
        max_size = cfg.TEST.BBOX_AUG.MAX_SIZE
        scores_scl, boxes_scl = im_detect_bbox_scale(
            model, im, scale, max_size, box_proposals)
        add_preds_t(scores_scl, boxes_scl)

        if cfg.TEST.BBOX_AUG.SCALE_H_FLIP:
            scores_scl_hf, boxes_scl_hf = im_detect_bbox_scale(
                model, im, scale, max_size, box_proposals, hflip=True)
            add_preds_t(scores_scl_hf, boxes_scl_hf)

    # Perform detection at different aspect ratios
    for aspect_ratio in cfg.TEST.BBOX_AUG.ASPECT_RATIOS:
        scores_ar, boxes_ar = im_detect_bbox_aspect_ratio(
            model, im, aspect_ratio, box_proposals)
        add_preds_t(scores_ar, boxes_ar)

        if cfg.TEST.BBOX_AUG.ASPECT_RATIO_H_FLIP:
            scores_ar_hf, boxes_ar_hf = im_detect_bbox_aspect_ratio(
                model, im, aspect_ratio, box_proposals, hflip=True)
            add_preds_t(scores_ar_hf, boxes_ar_hf)

    # Compute detections for the original image (identity transform) last to
    # ensure that the Caffe2 workspace is populated with blobs corresponding
    # to the original image on return (postcondition of im_detect_bbox)
    scores_i, boxes_i, im_scales_i = im_detect_bbox(model, im, box_proposals)
    add_preds_t(scores_i, boxes_i)

    # Combine the predicted scores
    if cfg.TEST.BBOX_AUG.SCORE_HEUR == 'ID':
        scores_c = scores_i
    elif cfg.TEST.BBOX_AUG.SCORE_HEUR == 'AVG':
        scores_c = np.mean(scores_ts, axis=0)
    elif cfg.TEST.BBOX_AUG.SCORE_HEUR == 'UNION':
        scores_c = np.vstack(scores_ts)
    else:
        raise NotImplementedError(
            'Score heur {} not supported'.format(cfg.TEST.BBOX_AUG.SCORE_HEUR))

    # Combine the predicted boxes
    if cfg.TEST.BBOX_AUG.COORD_HEUR == 'ID':
        boxes_c = boxes_i
    elif cfg.TEST.BBOX_AUG.COORD_HEUR == 'AVG':
        boxes_c = np.mean(boxes_ts, axis=0)
    elif cfg.TEST.BBOX_AUG.COORD_HEUR == 'UNION':
        boxes_c = np.vstack(boxes_ts)
    else:
        raise NotImplementedError(
            'Coord heur {} not supported'.format(cfg.TEST.BBOX_AUG.COORD_HEUR))

    return scores_c, boxes_c, im_scales_i


def im_detect_mask(model, im_scales, boxes):
    """Infer instance segmentation masks. This function must be called after
    im_detect_bbox as it assumes that the Caffe2 workspace is already populated
    with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scales (list): image blob scales as returned by im_detect_bbox
        boxes (ndarray): R x 4 array of bounding box detections (e.g., as
            returned by im_detect_bbox)

    Returns:
        pred_masks (ndarray): R x K x M x M array of class specific soft masks
            output by the network (must be processed by segm_results to convert
            into hard masks in the original image coordinate space)
    """
    assert len(im_scales) == 1, \
        'Only single-image / single-scale batch implemented'

    M = cfg.MRCNN.RESOLUTION
    if boxes.shape[0] == 0:
        pred_masks = np.zeros((0, M, M), np.float32)
        return pred_masks

    inputs = {'mask_rois': _get_rois_blob(boxes, im_scales)}
    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois_for_test(inputs, 'mask_rois')

    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v)
    workspace.RunNet(model.mask_net.Proto().name)

    # Fetch masks
    pred_masks = workspace.FetchBlob(
        core.ScopedName('mask_fcn_probs')).squeeze()

    if cfg.MRCNN.CLS_SPECIFIC_MASK:
        pred_masks = pred_masks.reshape([-1, cfg.MODEL.NUM_CLASSES, M, M])
    else:
        pred_masks = pred_masks.reshape([-1, 1, M, M])

    return pred_masks


def im_detect_mask_hflip(model, im, boxes):
    """Performs mask detection on the horizontally flipped image.
    Function signature is the same as for im_detect_mask_aug.
    """
    # Compute the masks for the flipped image
    im_hf = im[:, ::-1, :]
    boxes_hf = box_utils.flip_boxes(boxes, im.shape[1])

    im_scales = im_conv_body_only(model, im_hf)
    masks_hf = im_detect_mask(model, im_scales, boxes_hf)

    # Invert the predicted soft masks
    masks_inv = masks_hf[:, :, :, ::-1]

    return masks_inv


def im_detect_mask_scale(model, im, scale, max_size, boxes, hflip=False):
    """Computes masks at the given scale."""

    # Remember the original scale
    orig_scales = cfg.TEST.SCALES
    orig_max_size = cfg.TEST.MAX_SIZE

    # Perform mask detection at the given scale
    cfg.TEST.SCALES = (scale, )
    cfg.TEST.MAX_SIZE = max_size

    if hflip:
        masks_scl = im_detect_mask_hflip(model, im, boxes)
    else:
        im_scales = im_conv_body_only(model, im)
        masks_scl = im_detect_mask(model, im_scales, boxes)

    # Restore the original scale
    cfg.TEST.SCALES = orig_scales
    cfg.TEST.MAX_SIZE = orig_max_size

    return masks_scl


def im_detect_mask_aspect_ratio(model, im, aspect_ratio, boxes, hflip=False):
    """Computes mask detections at the given width-relative aspect ratio."""

    # Perform mask detection on the transformed image
    im_ar = image_utils.aspect_ratio_rel(im, aspect_ratio)
    boxes_ar = box_utils.aspect_ratio(boxes, aspect_ratio)

    if hflip:
        masks_ar = im_detect_mask_hflip(model, im_ar, boxes_ar)
    else:
        im_scales = im_conv_body_only(model, im_ar)
        masks_ar = im_detect_mask(model, im_scales, boxes_ar)

    return masks_ar


def im_detect_mask_aug(model, im, boxes):
    """Performs mask detection with test-time augmentations.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im (ndarray): BGR image to test
        boxes (ndarray): R x 4 array of bounding boxes

    Returns:
        masks (ndarray): R x K x M x M array of class specific soft masks
    """
    assert not cfg.TEST.MASK_AUG.SCALE_SIZE_DEP, \
        'Size dependent scaling not implemented'

    # Collect masks computed under different transformations
    masks_ts = []

    # Compute masks for the original image (identity transform)
    im_scales_i = im_conv_body_only(model, im)
    masks_i = im_detect_mask(model, im_scales_i, boxes)
    masks_ts.append(masks_i)

    # Perform mask detection on the horizontally flipped image
    if cfg.TEST.MASK_AUG.H_FLIP:
        masks_hf = im_detect_mask_hflip(model, im, boxes)
        masks_ts.append(masks_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.MASK_AUG.SCALES:
        max_size = cfg.TEST.MASK_AUG.MAX_SIZE
        masks_scl = im_detect_mask_scale(model, im, scale, max_size, boxes)
        masks_ts.append(masks_scl)

        if cfg.TEST.MASK_AUG.SCALE_H_FLIP:
            masks_scl_hf = im_detect_mask_scale(
                model, im, scale, max_size, boxes, hflip=True)
            masks_ts.append(masks_scl_hf)

    # Compute masks at different aspect ratios
    for aspect_ratio in cfg.TEST.MASK_AUG.ASPECT_RATIOS:
        masks_ar = im_detect_mask_aspect_ratio(model, im, aspect_ratio, boxes)
        masks_ts.append(masks_ar)

        if cfg.TEST.MASK_AUG.ASPECT_RATIO_H_FLIP:
            masks_ar_hf = im_detect_mask_aspect_ratio(
                model, im, aspect_ratio, boxes, hflip=True)
            masks_ts.append(masks_ar_hf)

    # Combine the predicted soft masks
    if cfg.TEST.MASK_AUG.HEUR == 'SOFT_AVG':
        masks_c = np.mean(masks_ts, axis=0)
    elif cfg.TEST.MASK_AUG.HEUR == 'SOFT_MAX':
        masks_c = np.amax(masks_ts, axis=0)
    elif cfg.TEST.MASK_AUG.HEUR == 'LOGIT_AVG':
        def logit(y):
            return -1.0 * np.log((1.0 - y) / np.maximum(y, 1e-20))
        logit_masks = [logit(y) for y in masks_ts]
        logit_masks = np.mean(logit_masks, axis=0)
        masks_c = 1.0 / (1.0 + np.exp(-logit_masks))
    else:
        raise NotImplementedError(
            'Heuristic {} not supported'.format(cfg.TEST.MASK_AUG.HEUR))

    return masks_c


def im_detect_keypoints(model, im_scales, boxes):
    """Infer instance keypoint poses. This function must be called after
    im_detect_bbox as it assumes that the Caffe2 workspace is already populated
    with the necessary blobs.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im_scales (list): image blob scales as returned by im_detect_bbox
        boxes (ndarray): R x 4 array of bounding box detections (e.g., as
            returned by im_detect_bbox)

    Returns:
        pred_heatmaps (ndarray): R x J x M x M array of keypoint location
            logits (softmax inputs) for each of the J keypoint types output
            by the network (must be processed by keypoint_results to convert
            into point predictions in the original image coordinate space)
    """
    assert len(im_scales) == 1, \
        'Only single-image / single-scale batch implemented'
    time_dim = boxes.shape[-1] // 4

    M = cfg.KRCNN.HEATMAP_SIZE
    if boxes.shape[0] == 0:
        pred_heatmaps = np.zeros(
            (0, time_dim * cfg.KRCNN.NUM_KEYPOINTS, M, M), np.float32)
        return pred_heatmaps

    inputs = {'keypoint_rois': _get_rois_blob(boxes, im_scales)}

    # Add multi-level rois for FPN
    if cfg.FPN.MULTILEVEL_ROIS:
        _add_multilevel_rois_for_test(inputs, 'keypoint_rois')

    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v)
    workspace.RunNet(model.keypoint_net.Proto().name)

    pred_heatmaps = workspace.FetchBlob(core.ScopedName('kps_score')).squeeze()

    # In case of 1
    if pred_heatmaps.ndim == 3:
        pred_heatmaps = np.expand_dims(pred_heatmaps, axis=0)

    return pred_heatmaps


def im_detect_keypoints_hflip(model, im, boxes):
    """Computes keypoint predictions on the horizontally flipped image.
    Function signature is the same as for im_detect_keypoints_aug.
    """
    # Compute keypoints for the flipped image
    im_hf = im[:, ::-1, :]
    boxes_hf = box_utils.flip_boxes(boxes, im.shape[1])

    im_scales = im_conv_body_only(model, im_hf)
    heatmaps_hf = im_detect_keypoints(model, im_scales, boxes_hf)

    # Invert the predicted keypoints
    heatmaps_inv = keypoint_utils.flip_heatmaps(heatmaps_hf)

    return heatmaps_inv


def im_detect_keypoints_scale(model, im, scale, max_size, boxes, hflip=False):
    """Computes keypoint predictions at the given scale."""

    # Store the original scale
    orig_scales = cfg.TEST.SCALES
    orig_max_size = cfg.TEST.MAX_SIZE

    # Perform detection at the given scale
    cfg.TEST.SCALES = (scale, )
    cfg.TEST.MAX_SIZE = max_size

    if hflip:
        heatmaps_scl = im_detect_keypoints_hflip(model, im, boxes)
    else:
        im_scales = im_conv_body_only(model, im)
        heatmaps_scl = im_detect_keypoints(model, im_scales, boxes)

    # Restore the original scale
    cfg.TEST.SCALES = orig_scales
    cfg.TEST.MAX_SIZE = orig_max_size

    return heatmaps_scl


def im_detect_keypoints_aspect_ratio(
        model, im, aspect_ratio, boxes, hflip=False):
    """Detects keypoints at the given width-relative aspect ratio."""

    # Perform keypoint detectionon the transformed image
    im_ar = image_utils.aspect_ratio_rel(im, aspect_ratio)
    boxes_ar = box_utils.aspect_ratio(boxes, aspect_ratio)

    if hflip:
        heatmaps_ar = im_detect_keypoints_hflip(model, im_ar, boxes_ar)
    else:
        im_scales = im_conv_body_only(model, im_ar)
        heatmaps_ar = im_detect_keypoints(model, im_scales, boxes_ar)

    return heatmaps_ar


def im_detect_keypoints_aug(model, im, boxes):
    """Computes keypoint predictions with test-time augmentations.

    Arguments:
        model (DetectionModelHelper): the detection model to use
        im (ndarray): BGR image to test
        boxes (ndarray): R x 4 array of bounding boxes

    Returns:
        heatmaps (ndarray): R x J x M x M array of keypoint location logits
    """
    assert not cfg.TEST.KPS_AUG.SCALE_SIZE_DEP, \
        'Size dependent scaling not implemented'

    # Collect heatmaps predicted under different transformations
    heatmaps_ts = []

    # Compute the heatmaps for the original image (identity transform)
    im_scales = im_conv_body_only(model, im)
    heatmaps_i = im_detect_keypoints(model, im_scales, boxes)
    heatmaps_ts.append(heatmaps_i)

    # Perform keypoints detection on the horizontally flipped image
    if cfg.TEST.KPS_AUG.H_FLIP:
        heatmaps_hf = im_detect_keypoints_hflip(model, im, boxes)
        heatmaps_ts.append(heatmaps_hf)

    # Compute detections at different scales
    for scale in cfg.TEST.KPS_AUG.SCALES:
        max_size = cfg.TEST.KPS_AUG.MAX_SIZE
        heatmaps_scl = im_detect_keypoints_scale(
            model, im, scale, max_size, boxes)
        heatmaps_ts.append(heatmaps_scl)

        if cfg.TEST.KPS_AUG.SCALE_H_FLIP:
            heatmaps_scl_hf = im_detect_keypoints_scale(
                model, im, scale, max_size, boxes, hflip=True)
            heatmaps_ts.append(heatmaps_scl_hf)

    # Compute keypoints at different aspect ratios
    for aspect_ratio in cfg.TEST.KPS_AUG.ASPECT_RATIOS:
        heatmaps_ar = im_detect_keypoints_aspect_ratio(
            model, im, aspect_ratio, boxes)
        heatmaps_ts.append(heatmaps_ar)

        if cfg.TEST.KPS_AUG.ASPECT_RATIO_H_FLIP:
            heatmaps_ar_hf = im_detect_keypoints_aspect_ratio(
                model, im, aspect_ratio, boxes, hflip=True)
            heatmaps_ts.append(heatmaps_ar_hf)

    # Combine the predicted heatmaps
    if cfg.TEST.KPS_AUG.HEUR == 'HM_AVG':
        heatmaps_c = np.mean(heatmaps_ts, axis=0)
    elif cfg.TEST.KPS_AUG.HEUR == 'HM_MAX':
        heatmaps_c = np.amax(heatmaps_ts, axis=0)
    else:
        raise NotImplementedError(
            'Heuristic {} not supported'.format(cfg.TEST.KPS_AUG.HEUR))

    return heatmaps_c


def box_results_with_nms_and_limit(scores, boxes):
    """Returns bounding-box detection results by thresholding on scores and
    applying non-maximum suppression (NMS).

    `boxes` has shape (#detections, 4 * #classes), where each row represents
    a list of predicted bounding boxes for each of the object classes in the
    dataset (including the background class). The detections in each row
    originate from the same object proposal.

    `scores` has shape (#detection, #classes), where each row represents a list
    of object detection confidence scores for each of the object classes in the
    dataset (including the background class). `scores[i, j]`` corresponds to the
    box at `boxes[i, j * 4:(j + 1) * 4]`.
    """
    num_classes = cfg.MODEL.NUM_CLASSES
    time_dim = boxes.shape[-1] // (num_classes * 4)
    cls_boxes = [[] for _ in range(num_classes)]
    # Apply threshold on detection probabilities and apply NMS
    # Skip j = 0, because it's the background class
    for j in range(1, num_classes):
        inds = np.where(scores[:, j] > cfg.TEST.SCORE_THRESH)[0]
        scores_j = scores[inds, j]
        boxes_j = boxes[inds, j * 4 * time_dim:(j + 1) * 4 * time_dim]
        dets_j = np.hstack((boxes_j, scores_j[:, np.newaxis])).astype(
            np.float32, copy=False)
        if cfg.TEST.SOFT_NMS.ENABLED:
            # Not implemented for time_dim > 1
            nms_dets = soft_nms(
                dets_j,
                sigma=cfg.TEST.SOFT_NMS.SIGMA,
                overlap_thresh=cfg.TEST.NMS,
                score_thresh=0.0001,
                method=cfg.TEST.SOFT_NMS.METHOD)
        else:
            keep = nms(dets_j, cfg.TEST.NMS)
            nms_dets = dets_j[keep, :]
        # Refine the post-NMS boxes using bounding-box voting
        if cfg.TEST.BBOX_VOTE.ENABLED:
            nms_dets = box_utils.box_voting(
                nms_dets, dets_j, cfg.TEST.BBOX_VOTE.VOTE_TH)
        cls_boxes[j] = nms_dets

    # Limit to max_per_image detections **over all classes**
    if cfg.TEST.DETECTIONS_PER_IM > 0:
        image_scores = np.hstack(
            [cls_boxes[j][:, -1] for j in range(1, num_classes)])
        if len(image_scores) > cfg.TEST.DETECTIONS_PER_IM:
            image_thresh = np.sort(
                image_scores)[-cfg.TEST.DETECTIONS_PER_IM]
            for j in range(1, num_classes):
                keep = np.where(cls_boxes[j][:, -1] >= image_thresh)[0]
                cls_boxes[j] = cls_boxes[j][keep, :]

    im_results = np.vstack([cls_boxes[j] for j in range(1, num_classes)])
    boxes = im_results[:, :-1]
    scores = im_results[:, -1]
    return scores, boxes, cls_boxes


def segm_results(cls_boxes, masks, ref_boxes, im_h, im_w):
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_segms = [[] for _ in range(num_classes)]
    mask_ind = 0
    # To work around an issue with cv2.resize (it seems to automatically pad
    # with repeated border values), we manually zero-pad the masks by 1 pixel
    # prior to resizing back to the original image resolution. This prevents
    # "top hat" artifacts. We therefore need to expand the reference boxes by an
    # appropriate factor.
    M = cfg.MRCNN.RESOLUTION
    scale = (M + 2.0) / M
    ref_boxes = box_utils.expand_boxes(ref_boxes, scale)
    ref_boxes = ref_boxes.astype(np.int32)
    padded_mask = np.zeros((M + 2, M + 2), dtype=np.float32)

    # skip j = 0, because it's the background class
    for j in range(1, num_classes):
        segms = []
        for _ in range(cls_boxes[j].shape[0]):
            if cfg.MRCNN.CLS_SPECIFIC_MASK:
                padded_mask[1:-1, 1:-1] = masks[mask_ind, j, :, :]
            else:
                padded_mask[1:-1, 1:-1] = masks[mask_ind, 0, :, :]

            ref_box = ref_boxes[mask_ind, :]
            w = ref_box[2] - ref_box[0] + 1
            h = ref_box[3] - ref_box[1] + 1
            w = np.maximum(w, 1)
            h = np.maximum(h, 1)

            mask = cv2.resize(padded_mask, (w, h))
            mask = np.array(mask > cfg.MRCNN.THRESH_BINARIZE, dtype=np.uint8)
            im_mask = np.zeros((im_h, im_w), dtype=np.uint8)

            x_0 = max(ref_box[0], 0)
            x_1 = min(ref_box[2] + 1, im_w)
            y_0 = max(ref_box[1], 0)
            y_1 = min(ref_box[3] + 1, im_h)

            im_mask[y_0:y_1, x_0:x_1] = mask[
                (y_0 - ref_box[1]):(y_1 - ref_box[1]),
                (x_0 - ref_box[0]):(x_1 - ref_box[0])]

            # Get RLE encoding used by the COCO evaluation API
            rle = mask_util.encode(
                np.array(im_mask[:, :, np.newaxis], order='F'))[0]
            segms.append(rle)

            mask_ind += 1

        cls_segms[j] = segms

    assert mask_ind == masks.shape[0]
    return cls_segms


def keypoint_results(cls_boxes, pred_heatmaps, ref_boxes):
    num_classes = cfg.MODEL.NUM_CLASSES
    cls_keyps = [[] for _ in range(num_classes)]
    person_idx = keypoint_utils.get_person_class_index()

    # handle the tubes
    assert pred_heatmaps.shape[1] % cfg.KRCNN.NUM_KEYPOINTS == 0, \
        'Heatmaps must be 17xT'
    time_dim = pred_heatmaps.shape[1] // cfg.KRCNN.NUM_KEYPOINTS
    assert time_dim == ref_boxes.shape[-1] // 4, 'Same T for boxes and keypoints'
    all_xy_preds = []
    for t in range(time_dim):
        all_xy_preds.append(keypoint_utils.heatmaps_to_keypoints(
            pred_heatmaps[:, t * cfg.KRCNN.NUM_KEYPOINTS:
                          (t + 1) * cfg.KRCNN.NUM_KEYPOINTS, ...],
            ref_boxes[:, t * 4: (t + 1) * 4]))
    xy_preds = np.concatenate(all_xy_preds, axis=-1)

    # NMS OKS
    if cfg.KRCNN.NMS_OKS:
        raise NotImplementedError('Handle tubes')
        keep = keypoint_utils.nms_oks(xy_preds, ref_boxes, 0.3)
        xy_preds = xy_preds[keep, :, :]
        ref_boxes = ref_boxes[keep, :]
        pred_heatmaps = pred_heatmaps[keep, :, :, :]
        cls_boxes[person_idx] = cls_boxes[person_idx][keep, :]

    kps = [xy_preds[i] for i in range(xy_preds.shape[0])]
    cls_keyps[person_idx] = kps
    return cls_keyps


def im_detect_all(model, im, box_proposals, timers=None):
    if timers is None:
        timers = defaultdict(Timer)

    timers['im_detect_bbox'].tic()
    if cfg.TEST.COMPETITION_MODE:
        scores, boxes, im_scales = im_detect_bbox_aug(model, im, box_proposals)
    else:
        scores, boxes, im_scales = im_detect_bbox(model, im, box_proposals)
    timers['im_detect_bbox'].toc()

    # score and boxes are from the whole image after score thresholding and nms
    # (they are not separated by class)
    # cls_boxes boxes and scores are separated by class and in the format used
    # for evaluating results
    timers['misc_bbox'].tic()
    scores, boxes, cls_boxes = box_results_with_nms_and_limit(scores, boxes)
    timers['misc_bbox'].toc()

    if cfg.MODEL.MASK_ON and boxes.shape[0] > 0:
        raise NotImplementedError('Handle tubes..')
        timers['im_detect_mask'].tic()
        if cfg.TEST.COMPETITION_MODE:
            masks = im_detect_mask_aug(model, im, boxes)
        else:
            masks = im_detect_mask(model, im_scales, boxes)
        timers['im_detect_mask'].toc()

        timers['misc_mask'].tic()
        cls_segms = segm_results(
            cls_boxes, masks, boxes, im.shape[0], im.shape[1])
        timers['misc_mask'].toc()
    else:
        cls_segms = None

    if cfg.MODEL.KEYPOINTS_ON and boxes.shape[0] > 0:
        timers['im_detect_keypoints'].tic()
        if cfg.TEST.COMPETITION_MODE:
            heatmaps = im_detect_keypoints_aug(model, im, boxes)
        else:
            heatmaps = im_detect_keypoints(model, im_scales, boxes)
        timers['im_detect_keypoints'].toc()

        timers['misc_keypoints'].tic()
        cls_keyps = keypoint_results(cls_boxes, heatmaps, boxes)
        timers['misc_keypoints'].toc()
    else:
        cls_keyps = None

    # Debugging
    # from utils import vis
    # T = vis.vis_one_image_opencv(im[0], cls_boxes[1], keypoints=cls_keyps[1])
    # time_dim = (cls_boxes[1].shape[1] - 1) // 4
    # for t in range(time_dim):
    #     T = vis.vis_one_image_opencv(
    #         im[t],
    #         cls_boxes[1][:, range(t * 4, (t + 1) * 4) + [-1]],
    #         keypoints=[el[:, 17 * t: (t + 1) * 17] for el in cls_keyps[1]])
    #     cv2.imwrite('/tmp/{}.jpg'.format(t + 1), T)
    # import pdb; pdb.set_trace()

    return cls_boxes, cls_segms, cls_keyps


def im_conv_body_only(model, im):
    """Runs `model.conv_body_net` on the given image `im`."""
    im_blob, im_scale_factors = _get_image_blob(im)
    workspace.FeedBlob(core.ScopedName('data'), im_blob)
    workspace.RunNet(model.conv_body_net.Proto().name)
    return im_scale_factors
