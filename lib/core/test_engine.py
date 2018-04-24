##############################################################
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################

"""Test a Fast R-CNN network on an imdb (image database)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import cv2
import os
import yaml
import datetime
from collections import defaultdict

from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace

from core.config import cfg, get_output_dir
from core.test import im_detect_all
from utils.timer import Timer
import utils.vis as vis_utils
import utils.subprocess as subprocess_utils
from datasets.json_dataset import JsonDataset
import utils.net as net_utils
from utils.io import robust_pickle_dump
from modeling import model_builder
import utils.image as image_utils
import utils.video as video_utils
from core.tracking_engine import run_posetrack_tracking

import logging
logger = logging.getLogger(__name__)

# OpenCL is enabled by default in OpenCV3 and it is not thread-safe leading
# to huge GPU memory allocations.
try:
    cv2.ocl.setUseOpenCL(False)
except AttributeError:
    pass


def initialize_model_from_cfg():
    def create_input_blobs(net_def):
        for op in net_def.op:
            for blob_in in op.input:
                if not workspace.HasBlob(blob_in):
                    workspace.CreateBlob(blob_in)

    model = model_builder.create(
        cfg.MODEL.TYPE, train=False,
        init_params=cfg.TEST.INIT_RANDOM_VARS_BEFORE_LOADING)
    model_builder.add_inputs(model)
    if cfg.TEST.INIT_RANDOM_VARS_BEFORE_LOADING:
        workspace.RunNetOnce(model.param_init_net)
    net_utils.initialize_from_weights_file(
        model, cfg.TEST.WEIGHTS, broadcast=False)
    create_input_blobs(model.net.Proto())
    workspace.CreateNet(model.net)
    workspace.CreateNet(model.conv_body_net)
    if cfg.MODEL.MASK_ON:
        create_input_blobs(model.mask_net.Proto())
        workspace.CreateNet(model.mask_net)
    if cfg.MODEL.KEYPOINTS_ON:
        create_input_blobs(model.keypoint_net.Proto())
        workspace.CreateNet(model.keypoint_net)
    return model


def get_roidb_and_dataset(ind_range, include_gt=False):
    """
    include_gt is used by the eval_mpii code. Not here.
    """
    dataset = JsonDataset(cfg.TEST.DATASET)
    if cfg.MODEL.FASTER_RCNN:
        roidb = dataset.get_roidb(gt=include_gt)
    else:
        roidb = dataset.get_roidb(
            gt=include_gt,
            proposal_file=cfg.TEST.PROPOSAL_FILE,
            proposal_limit=cfg.TEST.PROPOSAL_LIMIT)

    # Video processing (same as datasets/roidb.py)
    if cfg.MODEL.VIDEO_ON:
        roidb = video_utils.get_clip(roidb, remove_imperfect=False)

    if ind_range is not None:
        total_num_images = len(roidb)
        start, end = ind_range
        roidb = roidb[start:end]
    else:
        start = 0
        end = len(roidb)
        total_num_images = end

    return roidb, dataset, start, end, total_num_images


def empty_results(num_classes, num_images):
    # all detections are collected into:
    #    all_boxes[cls][image] = N x 5 array of detections in
    #    (x1, y1, x2, y2, score)
    all_boxes = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]
    all_segms = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]
    all_keyps = [[[] for _ in range(num_images)]
                 for _ in range(num_classes)]
    return all_boxes, all_segms, all_keyps


def extend_results(index, all_res, im_res):
    for j in range(1, len(im_res)):
        all_res[j][index] = im_res[j]


def test_net(ind_range=None):
    assert cfg.TEST.WEIGHTS != '', \
        'TEST.WEIGHTS must be set to the model file to test'
    assert not cfg.MODEL.RPN_ONLY, \
        'Use rpn_generate to generate proposals from RPN-only models'
    assert cfg.TEST.DATASET != '', \
        'TEST.DATASET must be set to the dataset name to test'

    output_dir = get_output_dir(training=False)
    roidb, dataset, start_ind, end_ind, total_num_images = \
        get_roidb_and_dataset(ind_range)
    model = initialize_model_from_cfg()
    num_images = len(roidb)
    num_classes = cfg.MODEL.NUM_CLASSES
    all_boxes, all_segms, all_keyps = empty_results(num_classes, num_images)
    timers = defaultdict(Timer)
    gpu_dev = core.DeviceOption(caffe2_pb2.CUDA, cfg.ROOT_GPU_ID)
    name_scope = 'gpu_{}'.format(cfg.ROOT_GPU_ID)
    for i, entry in enumerate(roidb):
        if cfg.MODEL.FASTER_RCNN:
            box_proposals = None
        else:
            # The roidb may contain ground-truth rois (for example, if the roidb
            # comes from the training or val split). We only want to evaluate
            # detection on the *non*-ground-truth rois. We select only the rois
            # that have the gt_classes field set to 0, which means there's no
            # ground truth.
            box_proposals = entry['boxes'][entry['gt_classes'] == 0]
            if len(box_proposals) == 0:
                continue

        im = image_utils.read_image_video(entry)
        with core.NameScope(name_scope):
            with core.DeviceScope(gpu_dev):
                cls_boxes_i, cls_segms_i, cls_keyps_i = im_detect_all(
                    model, im, box_proposals, timers)

        extend_results(i, all_boxes, cls_boxes_i)
        if cls_segms_i is not None:
            extend_results(i, all_segms, cls_segms_i)
        if cls_keyps_i is not None:
            extend_results(i, all_keyps, cls_keyps_i)

        if i % 10 == 0:  # Reduce log file size
            ave_total_time = np.sum([t.average_time for t in timers.values()])
            eta_seconds = ave_total_time * (num_images - i - 1)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            det_time = (timers['im_detect_bbox'].average_time +
                        timers['im_detect_mask'].average_time +
                        timers['im_detect_keypoints'].average_time)
            misc_time = (timers['misc_bbox'].average_time +
                         timers['misc_mask'].average_time +
                         timers['misc_keypoints'].average_time)
            logger.info(
                ('im_detect: range [{:d}, {:d}] of {:d}: '
                 '{:d}/{:d} {:.3f}s + {:.3f}s (eta: {})').format(
                    start_ind + 1, end_ind, total_num_images,
                    start_ind + i + 1, start_ind + num_images,
                    det_time, misc_time, eta))

        if cfg.VIS:
            im_name = os.path.splitext(os.path.basename(entry['image']))[0]
            vis_utils.vis_one_image(
                im[:, :, ::-1], '{:d}_{:s}'.format(i, im_name),
                os.path.join(output_dir, 'vis'), cls_boxes_i,
                segms=cls_segms_i, keypoints=cls_keyps_i,
                thresh=cfg.VIS_THR,
                box_alpha=0.8, dataset=dataset, show_class=True)

    cfg_yaml = yaml.dump(cfg)
    if ind_range is not None:
        det_name = 'detection_range_%s_%s.pkl' % tuple(ind_range)
    else:
        det_name = 'detections.pkl'
    det_file = os.path.join(output_dir, det_name)
    robust_pickle_dump(
        dict(all_boxes=all_boxes,
             all_segms=all_segms,
             all_keyps=all_keyps,
             cfg=cfg_yaml),
        det_file)
    logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))
    return all_boxes, all_segms, all_keyps


def evaluate_detections(dataset, all_boxes, output_dir, use_matlab=False):
    logger.info('Evaluating detections')
    not_comp = not cfg.TEST.COMPETITION_MODE
    if dataset.name.find('coco_') > -1 or cfg.TEST.FORCE_JSON_DATASET_EVAL:
        import datasets.json_dataset_evaluator as json_dataset_evaluator
        json_dataset_evaluator.evaluate_detections(
            dataset, all_boxes, output_dir, use_salt=not_comp, cleanup=not_comp)
    elif dataset.name.find('cityscapes_') > -1:
        import datasets.json_dataset_evaluator as json_dataset_evaluator
        logger.warn('Cityscapes bbox evaluated using COCO metrics/conversions')
        json_dataset_evaluator.evaluate_detections(
            dataset, all_boxes, output_dir, use_salt=not_comp, cleanup=not_comp)
    elif dataset.name[:4] == 'voc_':
        # For VOC, always use salt and always cleanup because results are
        # written to the shared VOCdevkit results directory
        import datasets.voc_dataset_evaluator as voc_dataset_evaluator
        voc_dataset_evaluator.evaluate_detections(
            dataset, all_boxes, output_dir, use_matlab=use_matlab)
    else:
        raise NotImplementedError(
            'No evaluator for dataset: {}'.format(dataset.name))


def evaluate_segmentations(dataset, all_boxes, all_segms, output_dir):
    logger.info('Evaluating segmentations')
    not_comp = not cfg.TEST.COMPETITION_MODE
    if dataset.name.find('coco_') > -1 or cfg.TEST.FORCE_JSON_DATASET_EVAL:
        import datasets.json_dataset_evaluator as json_dataset_evaluator
        json_dataset_evaluator.evaluate_segmentations(
            dataset,
            all_boxes, all_segms,
            output_dir, use_salt=not_comp, cleanup=not_comp)
    elif dataset.name.find('cityscapes_') > -1:
        import datasets.cityscapes_json_dataset_evaluator \
            as cityscapes_json_dataset_evaluator
        cityscapes_json_dataset_evaluator.evaluate_segmentations(
            dataset,
            all_boxes, all_segms,
            output_dir, use_salt=not_comp, cleanup=not_comp)
    else:
        raise NotImplementedError(
            'No evaluator for dataset: {}'.format(dataset.name))


def evaluate_keypoints(dataset, all_boxes, all_keyps, output_dir):
    logger.info('Evaluating detections')
    not_comp = not cfg.TEST.COMPETITION_MODE
    if dataset.name.startswith('keypoints_coco_'):
        import datasets.json_dataset_evaluator as json_dataset_evaluator
        json_dataset_evaluator.evaluate_keypoints(
            dataset, all_boxes, all_keyps, output_dir,
            use_salt=not_comp, cleanup=not_comp)
    else:
        raise NotImplementedError(
            'No evaluator for dataset: {}'.format(dataset.name))


def evaluate_all(
        dataset, all_boxes, all_segms, all_keyps, output_dir, use_matlab=False):
    evaluate_detections(dataset, all_boxes, output_dir, use_matlab=use_matlab)
    logger.info('Evaluating bounding boxes is done!')
    if cfg.MODEL.MASK_ON:
        evaluate_segmentations(dataset, all_boxes, all_segms, output_dir)
        logger.info('Evaluating segmentations is done!')
    if cfg.MODEL.KEYPOINTS_ON:
        evaluate_keypoints(dataset, all_boxes, all_keyps, output_dir)
        logger.info('Evaluating keypoints is done!')


def multi_gpu_test_net_on_dataset(num_images, output_dir):
    binary = os.path.join('tools/test_net.py')
    assert os.path.exists(binary), 'Binary {} not found'.format(binary)

    # Run inference in parallel in subprocesses
    outputs = subprocess_utils.process_in_parallel(
        'detection', num_images, binary, output_dir)

    # Collate the results from each subprocess
    all_boxes = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    all_segms = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    all_keyps = [[] for _ in range(cfg.MODEL.NUM_CLASSES)]
    for det_data in outputs:
        all_boxes_batch = det_data['all_boxes']
        all_segms_batch = det_data['all_segms']
        all_keyps_batch = det_data['all_keyps']
        for j in range(1, cfg.MODEL.NUM_CLASSES):
            all_boxes[j] += all_boxes_batch[j]
            all_segms[j] += all_segms_batch[j]
            all_keyps[j] += all_keyps_batch[j]
    det_file = os.path.join(output_dir, 'detections.pkl')
    cfg_yaml = yaml.dump(cfg)
    robust_pickle_dump(
        dict(all_boxes=all_boxes,
             all_segms=all_segms,
             all_keyps=all_keyps,
             cfg=cfg_yaml),
        det_file)
    logger.info('Wrote detections to: {}'.format(os.path.abspath(det_file)))

    return all_boxes, all_segms, all_keyps


def test_net_on_dataset(multi_gpu=False):
    output_dir = get_output_dir(training=False)
    dataset = JsonDataset(cfg.TEST.DATASET)
    test_timer = Timer()
    test_timer.tic()
    if multi_gpu:
        num_images = len(dataset.get_roidb())
        all_boxes, all_segms, all_keyps = multi_gpu_test_net_on_dataset(
            num_images, output_dir)
    else:
        all_boxes, all_segms, all_keyps = test_net()
    test_timer.toc()
    logger.info('Total inference time: {:.3f}s'.format(
        test_timer.average_time))
    # Run tracking and eval for posetrack datasets
    if dataset.name.startswith('posetrack') or dataset.name.startswith('kinetics'):
        roidb, dataset, _, _, _ = get_roidb_and_dataset(None)
        run_posetrack_tracking(output_dir, roidb)
    try:
        evaluate_all(dataset, all_boxes, all_segms, all_keyps, output_dir)
    except Exception as e:
        # Typically would crash as we don't have evaluators for each dataset
        logger.error('Evaluation crashed with exception {}'.format(e))
