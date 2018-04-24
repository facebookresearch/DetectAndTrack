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

import argparse
import os.path as osp
import sys
import cPickle as pickle
import cv2
import logging
import numpy as np
from tqdm import tqdm

from core.test_engine import get_roidb_and_dataset
import utils.vis as vis_utils
import utils.image as image_utils
from core.config import (
    cfg_from_file, assert_and_infer_cfg, get_output_dir, cfg_from_list)
import utils.general as gen_utils

FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--cfg', dest='cfg_file', help='Config file', type=str)
    parser.add_argument(
        '--thresh', dest='thresh',
        help='detection prob threshold',
        default=0.9, type=float)
    parser.add_argument(
        'opts', help='See lib/core/config.py for all options', default=None,
        nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    args = parser.parse_args()
    return args


def _id_or_index(ix, val):
    if len(val) == 0:
        return val
    else:
        return val[ix]


def _vis_single_frame(im, cls_boxes_i, cls_segms_i, cls_keyps_i, cls_tracks_i, thresh):
    res = vis_utils.vis_one_image_opencv(
        im, cls_boxes_i,
        segms=cls_segms_i, keypoints=cls_keyps_i,
        tracks=cls_tracks_i, thresh=thresh,
        show_box=True, show_class=False, linewidth=3)
    if res is None:
        return im
    return res


def _convert_roidb_to_pred_boxes(boxes):
    return np.hstack((boxes, np.ones((boxes.shape[0], 1))))


def _convert_roidb_to_pred_keyps(poses):
    poses = poses.astype(np.float32)
    res = []
    for i in range(poses.shape[0]):
        poses[i, 2, poses[i, 2, :] >= 2] += 10.0
        res.append(np.vstack((poses[i], np.zeros((1, poses[i].shape[1])))))
    return res


def _convert_roidb_to_pred_tracks(tracks):
    return tracks.reshape((-1, )).tolist()


def _generate_visualizations(entry, ix, all_boxes, all_keyps, all_tracks, thresh):
    im = image_utils.read_image_video(entry, key_frame_only=True)[0]
    cls_boxes_i = [
        _id_or_index(ix, all_boxes[j]) for j in range(len(all_boxes))]
    if all_keyps is not None:
        cls_keyps_i = [
            _id_or_index(ix, all_keyps[j]) for j in range(len(all_keyps))]
    else:
        cls_keyps_i = None
    if all_tracks is not None:
        cls_tracks_i = [
            _id_or_index(ix, all_tracks[j]) for j in range(len(all_tracks))]
    else:
        cls_tracks_i = None
    pred = _vis_single_frame(
        im.copy(), cls_boxes_i, None, cls_keyps_i, cls_tracks_i, thresh)
    gt = _vis_single_frame(
        im.copy(),
        [[], _convert_roidb_to_pred_boxes(entry['boxes'])],
        None,
        [[], _convert_roidb_to_pred_keyps(entry['gt_keypoints'])],
        [[], _convert_roidb_to_pred_tracks(entry['tracks'])],
        0.1)
    return gt, pred


def vis(roidb, detections_pkl, thresh, output_dir):
    if len(roidb) == 0:
        return
    with open(detections_pkl, 'rb') as f:
        dets = pickle.load(f)

    all_boxes = dets['all_boxes']
    if 'all_keyps' in dets:
        all_keyps = dets['all_keyps']
    else:
        all_keyps = None
    if 'all_tracks' in dets:
        all_tracks = dets['all_tracks']
    else:
        all_tracks = None

    for ix, entry in enumerate(tqdm(roidb)):
        if entry['boxes'] is None or entry['boxes'].shape[0] == 0:
            continue
        gt, pred = _generate_visualizations(
            entry, ix, all_boxes, all_keyps, all_tracks, thresh)
        combined = np.hstack((gt, pred))
        im_name = entry['image']
        if isinstance(im_name, list):
            im_name = im_name[len(im_name) // 2]
        out_name = im_name[len(dataset.image_directory):]
        out_path = osp.join(output_dir, out_name)
        gen_utils.mkdir_p(osp.dirname(out_path))
        cv2.imwrite(out_path, combined)


if __name__ == '__main__':
    args = _parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.opts is not None:
        cfg_from_list(args.opts)
    assert_and_infer_cfg()
    test_output_dir = get_output_dir(training=False)
    det_file = osp.join(test_output_dir, 'detections.pkl')
    tracking_det_file = osp.join(test_output_dir, 'detections_withTracks.pkl')
    if osp.exists(tracking_det_file):
        det_file = tracking_det_file
    output_dir = osp.join(test_output_dir, 'vis/')
    if not osp.exists(det_file):
        raise ValueError('Output file not found {}'.format(det_file))
    else:
        logger.info('Visualizing {}'.format(det_file))
    # Set include_gt True when using the roidb to evalute directly. Not doing
    # that currently
    roidb, dataset, _, _, _ = get_roidb_and_dataset(None, include_gt=True)
    vis(roidb, det_file, args.thresh, output_dir)
