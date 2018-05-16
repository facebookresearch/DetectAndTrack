##############################################################
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################

# --------------------------------------------------------
# Faster R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from libfb.py import parutil

import numpy as np
import cv2
import datetime
import os
import yaml

from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace

from core.config import cfg, get_output_dir
from utils.blob import im_list_to_blob
from utils.timer import Timer
from utils.io import robust_pickle_dump
import utils.net as nu
import utils.subprocess as subprocess_utils
from datasets.json_dataset import JsonDataset
import datasets.json_dataset_evaluator as json_dataset_evaluator
from modeling import model_builder

import logging
logger = logging.getLogger(__name__)

# OpenCL is enabled by default in OpenCV3 and it is not thread-safe leading
# to huge GPU memory allocations. See https://fburl.com/9d7tvusd
cv2.ocl.setUseOpenCL(False)


def generate_rpn_on_dataset(multi_gpu=False):
    output_dir = get_output_dir(training=False)
    dataset = JsonDataset(cfg.TEST.DATASET)
    test_timer = Timer()
    test_timer.tic()
    if multi_gpu:
        num_images = len(dataset.get_roidb())
        _boxes, _scores, _ids, rpn_file = multi_gpu_generate_rpn_on_dataset(
            num_images, output_dir)
    else:
        # Processes entire dataset range by default
        _boxes, _scores, _ids, rpn_file = generate_rpn_on_range()
    test_timer.toc()
    logger.info('Total inference time: {:.3f}s'.format(
        test_timer.average_time))
    evaluate_proposal_file(dataset, rpn_file, output_dir)


def evaluate_proposal_file(dataset, proposal_file, output_dir):
    roidb = dataset.get_roidb(gt=True, proposal_file=proposal_file)
    logger.info('~~~~ Summary metrics ~~~~')
    recall_files = []
    for l in [300, 1000, 2000]:
        print(' res@{:d} proposals / image:'.format(l))
        res = {}
        for a in ['all', 'small', 'medium', 'large']:
            res[a] = json_dataset_evaluator.evaluate_recall(
                dataset, roidb, area=a, limit=l)
            print(' area={:8s} | ar={:.3f}'.format(a, res[a]['ar']))

        # Index 4 is for iou thresh of 0.7
        for a in ['all', 'small', 'medium', 'large']:
            print(
                ' iou=[.7]     | area={:8s} | ar={:.3f}'.
                format(a, res[a]['recalls'][4]))

        recall_file = os.path.join(
            output_dir, 'at{:d}'.format(l) + 'rpn_proposal_recall.pkl')
        robust_pickle_dump(res, recall_file)
        recall_files.append(recall_file)
    logger.info('Evaluating proposals is done!')
    return recall_files


def generate_rpn_on_range(ind_range=None):
    assert cfg.TEST.WEIGHTS != '', \
        'TEST.WEIGHTS must be set to the model file to test'
    assert cfg.TEST.DATASET != '', \
        'TEST.DATASET must be set to the dataset name to test'
    assert cfg.MODEL.RPN_ONLY or cfg.MODEL.FASTER_RCNN

    im_list, start_ind, end_ind, total_num_images = get_image_list(ind_range)
    output_dir = get_output_dir(training=False)
    logger.info(
        'Output will be saved to: {:s}'.format(os.path.abspath(output_dir)))

    model = model_builder.create(cfg.MODEL.TYPE, train=False)
    model_builder.add_inputs(model)
    nu.initialize_from_weights_file(model, cfg.TEST.WEIGHTS)
    workspace.CreateNet(model.net)

    boxes, scores, ids = im_list_proposals(
        model,
        im_list,
        start_ind=start_ind,
        end_ind=end_ind,
        total_num_images=total_num_images)

    cfg_yaml = yaml.dump(cfg)
    if ind_range is not None:
        rpn_name = 'rpn_proposals_range_%s_%s.pkl' % tuple(ind_range)
    else:
        rpn_name = 'rpn_proposals.pkl'
    rpn_file = os.path.join(output_dir, rpn_name)
    robust_pickle_dump(
        dict(boxes=boxes, scores=scores, ids=ids, cfg=cfg_yaml), rpn_file)
    logger.info('Wrote RPN proposals to {}'.format(os.path.abspath(rpn_file)))
    return boxes, scores, ids, rpn_file


def get_image_list(ind_range):
    dataset = JsonDataset(cfg.TEST.DATASET)
    roidb = dataset.get_roidb()

    if ind_range is not None:
        total_num_images = len(roidb)
        start, end = ind_range
        roidb = roidb[start:end]
    else:
        start = 0
        end = len(roidb)
        total_num_images = end

    return roidb, start, end, total_num_images


def multi_gpu_generate_rpn_on_dataset(num_images, output_dir):
    # TODO(rbg): Need to have non-FB specific code path for OSS
    if cfg.CLUSTER.ON_CLUSTER:
        binary_dir = os.path.abspath(os.getcwd())
        binary = os.path.join(binary_dir, 'test_net.xar')
    else:
        assert parutil.is_lpar(), 'Binary must be inplace package style'
        binary_dir = os.path.dirname(parutil.get_runtime_path())
        binary = os.path.join(binary_dir, 'test_net.par')
    assert os.path.exists(binary), 'Binary {} not found'.format(binary)

    # Run inference in parallel in subprocesses
    outputs = subprocess_utils.process_in_parallel(
        'rpn_proposals', num_images, binary, output_dir)

    # Collate the results from each subprocess
    boxes, scores, ids = [], [], []
    for rpn_data in outputs:
        boxes += rpn_data['boxes']
        scores += rpn_data['scores']
        ids += rpn_data['ids']
    rpn_file = os.path.join(output_dir, 'rpn_proposals.pkl')
    cfg_yaml = yaml.dump(cfg)
    robust_pickle_dump(
        dict(boxes=boxes, scores=scores, ids=ids, cfg=cfg_yaml), rpn_file)
    logger.info('Wrote RPN proposals to {}'.format(os.path.abspath(rpn_file)))
    return boxes, scores, ids, rpn_file


def im_proposals(model, im):
    """Generate RPN proposals on a single image."""
    inputs = {}
    inputs['data'], inputs['im_info'] = _get_image_blob(im)
    for k, v in inputs.items():
        workspace.FeedBlob(core.ScopedName(k), v.astype(np.float32, copy=False))
    workspace.RunNet(model.net.Proto().name)
    scale = inputs['im_info'][0, 2]

    if cfg.FPN.FPN_ON and cfg.FPN.MULTILEVEL_RPN:
        k_max = cfg.FPN.RPN_MAX_LEVEL
        k_min = cfg.FPN.RPN_MIN_LEVEL
        rois_names = [
            core.ScopedName('rpn_rois_fpn' + str(l))
            for l in range(k_min, k_max + 1)]
        score_names = [
            core.ScopedName('rpn_roi_probs_fpn' + str(l))
            for l in range(k_min, k_max + 1)]
        blobs = workspace.FetchBlobs(rois_names + score_names)
        # Combine predictions across all levels and retain the top scoring
        boxes = np.concatenate(blobs[:len(rois_names)])
        scores = np.concatenate(blobs[len(rois_names):]).squeeze()
        # TODO(rbg): NMS again?
        inds = np.argsort(-scores)[:cfg.TEST.RPN_POST_NMS_TOP_N]
        scores = scores[inds]
        boxes = boxes[inds, :]
    else:
        boxes, scores = workspace.FetchBlobs(
            [core.ScopedName('rpn_rois'), core.ScopedName('rpn_roi_probs')])
        scores = scores.squeeze()

    # Column 0 is the batch index in the (batch ind, x1, y1, x2, y2) encoding,
    # so we remove it since we just want to return boxes
    boxes = boxes[:, 1:] / scale
    return boxes, scores


def im_list_proposals(
    model, im_list, start_ind=None, end_ind=None, total_num_images=None
):
    """Generate RPN proposals on all images in an imdb."""

    _t = Timer()
    num_images = len(im_list)
    im_list_boxes = [[] for _ in range(num_images)]
    im_list_scores = [[] for _ in range(num_images)]
    im_list_ids = [[] for _ in range(num_images)]
    if start_ind is None:
        start_ind = 0
        end_ind = num_images
        total_num_images = num_images
    for i in range(num_images):
        im_list_ids[i] = im_list[i]['id']
        im = cv2.imread(im_list[i]['image'])
        with core.NameScope('gpu_{}'.format(cfg.ROOT_GPU_ID)):
            with core.DeviceScope(
                    core.DeviceOption(caffe2_pb2.CUDA, cfg.ROOT_GPU_ID)):
                _t.tic()
                im_list_boxes[i], im_list_scores[i] = im_proposals(model, im)
                _t.toc()
        if i % 10 == 0:
            ave_time = _t.average_time
            eta_seconds = ave_time * (num_images - i - 1)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            logger.info(
                ('rpn_generate: range [{:d}, {:d}] of {:d}: '
                 '{:d}/{:d} {:.3f}s (eta: {})').format(
                    start_ind + 1, end_ind, total_num_images,
                    start_ind + i + 1, start_ind + num_images,
                    ave_time, eta))

    return im_list_boxes, im_list_scores, im_list_ids


def _get_image_blob(im):
    """Converts an image into a network input.

    Arguments:
        im (ndarray): a color image in BGR order

    Returns:
        blob (ndarray): a data blob holding an image pyramid
        im_scale_factors (list): list of image scales (relative to im) used
            in the image pyramid
    """
    im_orig = im.astype(np.float32, copy=True)
    im_orig -= cfg.PIXEL_MEANS

    im_shape = im_orig.shape
    im_size_min = np.min(im_shape[0:2])
    im_size_max = np.max(im_shape[0:2])

    processed_ims = []

    assert len(cfg.TEST.SCALES) == 1
    target_size = cfg.TEST.SCALES[0]

    im_scale = float(target_size) / float(im_size_min)
    # Prevent the biggest axis from being more than MAX_SIZE
    if np.round(im_scale * im_size_max) > cfg.TEST.MAX_SIZE:
        im_scale = float(cfg.TEST.MAX_SIZE) / float(im_size_max)
    im = cv2.resize(im_orig, None, None, fx=im_scale, fy=im_scale,
                    interpolation=cv2.INTER_LINEAR)
    im_info = np.hstack((im.shape[:2], im_scale))[np.newaxis, :]
    processed_ims.append(im)

    # Create a blob to hold the input images
    blob = im_list_to_blob(processed_ims)

    return blob, im_info
