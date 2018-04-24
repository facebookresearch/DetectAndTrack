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

import logging
import sys
import numpy as np
import os
import os.path as osp
import cPickle as pkl
import json
from tqdm import tqdm
import scipy.io as sio
import shutil
import tempfile
from functools import partial
import time

from core.config import cfg
import utils.general as gen_utils
from utils.image import get_image_path

np.random.seed(cfg.RNG_SEED)
FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

coco_src_keypoints = [
    'nose',
    'left_eye',
    'right_eye',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle']
posetrack_src_keypoints = [
    'nose',
    'head_bottom',
    'head_top',
    'left_ear',
    'right_ear',
    'left_shoulder',
    'right_shoulder',
    'left_elbow',
    'right_elbow',
    'left_wrist',
    'right_wrist',
    'left_hip',
    'right_hip',
    'left_knee',
    'right_knee',
    'left_ankle',
    'right_ankle']
dst_keypoints = [
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
    'neck',
    'nose',
    'head_top']


def _compute_score(conf, global_conf):
    kp_conf_type = cfg.TRACKING.KP_CONF_TYPE
    if kp_conf_type == 'global':
        return global_conf
    elif kp_conf_type == 'local':
        return conf
    elif kp_conf_type == 'scaled':
        return conf * global_conf
    else:
        raise NotImplementedError('Uknown type {}'.format(kp_conf_type))


def coco2posetrack(preds, src_kps, dst_kps, global_score,
                   kp_conf_type=cfg.TRACKING.KP_CONF_TYPE):
    data = []
    global_score = float(global_score)
    dstK = len(dst_kps)
    for k in range(dstK):
        if dst_kps[k] in src_kps:
            ind = src_kps.index(dst_kps[k])
            local_score = (preds[2, ind] + preds[2, ind]) / 2.0
            conf = _compute_score(local_score, global_score)
            if local_score >= cfg.EVAL.EVAL_MPII_KPT_THRESHOLD:
                data.append({'id': [k],
                             'x': [float(preds[0, ind])],
                             'y': [float(preds[1, ind])],
                             'score': [conf]})
        elif dst_kps[k] == 'neck':
            rsho = src_kps.index('right_shoulder')
            lsho = src_kps.index('left_shoulder')
            x_msho = (preds[0, rsho] + preds[0, lsho]) / 2.0
            y_msho = (preds[1, rsho] + preds[1, lsho]) / 2.0
            local_score = (preds[2, rsho] + preds[2, lsho]) / 2.0
            conf_msho = _compute_score(local_score, global_score)
            if local_score >= cfg.EVAL.EVAL_MPII_KPT_THRESHOLD:
                data.append({'id': [k],
                             'x': [float(x_msho)],
                             'y': [float(y_msho)],
                             'score': [conf_msho]})
        elif dst_kps[k] == 'head_top':
            rsho = src_kps.index('right_shoulder')
            lsho = src_kps.index('left_shoulder')
            x_msho = (preds[0, rsho] + preds[0, lsho]) / 2.0
            y_msho = (preds[1, rsho] + preds[1, lsho]) / 2.0
            nose = src_kps.index('nose')
            x_nose = preds[0, nose]
            y_nose = preds[1, nose]
            x_tophead = x_nose - (x_msho - x_nose)
            y_tophead = y_nose - (y_msho - y_nose)
            local_score = (preds[2, rsho] + preds[2, lsho]) / 2.0
            conf_htop = _compute_score(local_score, global_score)
            if local_score >= cfg.EVAL.EVAL_MPII_KPT_THRESHOLD:
                data.append({
                    'id': [k],
                    'x': [float(x_tophead)],
                    'y': [float(y_tophead)],
                    'score': [conf_htop]})
    return data


def _convert_data_to_annorect_struct(boxes, poses, tracks):
    """
    Args:
        boxes (np.ndarray): Nx5 size matrix with boxes on this frame
        poses (list of np.ndarray): N length list with each element as 4x17 array
        tracks (list): N length list with track ID for each box/pose
    """
    num_dets = boxes.shape[0]
    annorect = []
    for j in range(num_dets):
        score = boxes[j, -1]
        if score < cfg.EVAL.EVAL_MPII_DROP_DETECTION_THRESHOLD:
            continue
        point = coco2posetrack(
            poses[j], posetrack_src_keypoints, dst_keypoints, score)
        annorect.append({'annopoints': [{'point': point}],
                         'score': [float(score)],
                         'track_id': [tracks[j]]})
    if num_dets == 0:
        # MOTA requires each image to have at least one detection! So, adding
        # a dummy prediction.
        annorect.append({
            'annopoints': [{'point': [{
                'id': [0],
                'x': [0],
                'y': [0],
                'score': [-100.0],
            }]}],
            'score': [0],
            'track_id': [0]})
    return annorect


def video2filenames(pathtodir):
    ext_types = '.mat'  # .mat/.json
    output = {}
    files = [f for f in os.listdir(pathtodir) if
             osp.isfile(osp.join(pathtodir, f)) and ext_types in f]
    for fname in files:
        if ext_types == '.mat':
            out_fname = fname.replace('.mat', '.json')
            data = sio.loadmat(
                osp.join(pathtodir, fname), squeeze_me=True,
                struct_as_record=False)
            temp = data['annolist'][0].image.name
        elif ext_types == '.json':
            out_fname = fname
            with open(osp.join(pathtodir, fname), 'r') as fin:
                data = json.load(fin)
            temp = data['annolist'][0]['image'][0]['name']
        else:
            raise NotImplementedError()
        video = osp.dirname(temp)
        output[video] = out_fname
    return output


def _run_eval(annot_dir, output_dir, eval_tracking=False, eval_pose=True):
    """
    Runs the evaluation, and returns the "total mAP" and "total MOTA"
    """
    from datasets.posetrack.poseval.py import evaluate_simple
    (apAll, _, _), mota = evaluate_simple.evaluate(
        annot_dir, output_dir, eval_pose, eval_tracking,
        cfg.TRACKING.DEBUG.UPPER_BOUND_4_EVAL_UPPER_BOUND)
    return apAll[-1][0], mota[-4][0]


def _run_eval_single_video(vname, out_filenames, output_dir, dataset, eval_tracking):
    per_vid_tmp_dir = tempfile.mkdtemp()
    gen_utils.mkdir_p(per_vid_tmp_dir)
    # in case it previously existed and has anything in it
    gen_utils.mkdir_p(osp.join(per_vid_tmp_dir, 'gt/'))
    gen_utils.mkdir_p(osp.join(per_vid_tmp_dir, 'pred/'))
    voutname = out_filenames[osp.join('images', vname)]
    pred_path = osp.join(
        output_dir, voutname)
    gt_path = osp.join(
        dataset.annotation_directory, voutname)
    shutil.copyfile(gt_path, osp.join(per_vid_tmp_dir, 'gt', voutname))
    shutil.copyfile(pred_path, osp.join(per_vid_tmp_dir, 'pred', voutname))
    try:
        score_ap, score_mot = _run_eval(
            osp.join(per_vid_tmp_dir, 'gt/'),
            osp.join(per_vid_tmp_dir, 'pred/'),
            eval_tracking)
    except Exception as e:
        logger.error('Unable to process video {} due to {}'.format(
            vname, e))
        score_ap = np.nan
        score_mot = np.nan
    gen_utils.run_cmd('rm -rf {}'.format(per_vid_tmp_dir), print_cmd=False)
    return (vname, score_ap, score_mot)


def _run_posetrack_eval(roidb, det_file, dataset, output_dir):
    with open(det_file, 'rb') as fin:
        dets = pkl.load(fin)
    assert len(roidb) == len(dets['all_boxes'][1]), \
        'Mismatch {} vs {}'.format(len(roidb), len(dets['all_boxes'][1]))
    gen_utils.mkdir_p(output_dir)
    out_filenames = video2filenames(dataset.annotation_directory)
    out_data = {}  # each video to all predictions
    eval_tracking = False
    if 'all_tracks' in dets:
        eval_tracking = True
    for i, entry in enumerate(roidb):
        image_name = get_image_path(entry)[len(dataset.image_directory):]
        video_name = osp.dirname(image_name)
        frame_num = int(osp.basename(image_name).split('.')[0])
        boxes = dets['all_boxes'][1][i]
        kps = dets['all_keyps'][1][i]
        if eval_tracking:  # means there is a "all_tracks" in the dets
            tracks = dets['all_tracks'][1][i]
        else:
            tracks = [1] * len(kps)
        data_el = {
            'image': image_name,
            'imagenum': [frame_num],
            'annorect': _convert_data_to_annorect_struct(boxes, kps, tracks),
        }
        if video_name in out_data:
            out_data[video_name].append(data_el)
        else:
            out_data[video_name] = [data_el]

    logger.info('Saving the JSON files to {}'.format(output_dir))
    # clear out the previous predictions, if any
    gen_utils.run_cmd('rm -r {}/*'.format(output_dir), print_cmd=False)
    for vname in tqdm(out_data.keys(), desc='Writing JSON files for eval'):
        vdata = out_data[vname]
        outfpath = osp.join(
            output_dir, out_filenames[osp.join('images', vname)])
        with open(outfpath, 'w') as fout:
            json.dump({'annolist': vdata}, fout)
    logger.info('Wrote all predictions in JSON to {}'.format(output_dir))
    logger.info('Running dataset level evaluation...')
    st_time = time.time()
    logger.info(_run_eval(dataset.annotation_directory, output_dir, eval_tracking))
    logger.info('...Done in {}'.format(time.time() - st_time))
    # TODO(rgirdhar): Do this better
    if cfg.EVAL.EVAL_MPII_PER_VIDEO:  # run the evaluation per-video
        res = []
        logger.info('Running per-video evaluation...')
        st_time = time.time()
        pervid_outpath = osp.join(
            osp.dirname(osp.normpath(output_dir)),
            osp.basename(det_file) + '_per_video_scores.txt')
        # Earlier I used multi-processing to compute the predictions in parallel
        # but now I've updated the eval code itself to use multiprocessing so
        # can not use multiprocessing here (else it gives an error that daemon
        # processes can not spawn children). Hense setting num processes to 0.
        res = map(partial(
            _run_eval_single_video,
            out_filenames=out_filenames,
            output_dir=output_dir,
            dataset=dataset,
            eval_tracking=eval_tracking), out_data.keys())
        logger.info('...Done in {} seconds'.format(time.time() - st_time))
        res = sorted(res, key=lambda x: x[1])  # sort on score
        logger.info('Writing per-video scores to {}'.format(pervid_outpath))
        with open(pervid_outpath, 'w') as fout:
            for el in res:
                fout.write('{} {} {}\n'.format(el[0], el[1], el[2]))


def run_mpii_eval(test_output_dir, roidb, dataset):
    # Set include_gt True when using the roidb to evalute directly. Not doing
    # that currently
    # det_file = osp.join(test_output_dir, 'detections.pkl')
    tracking_det_file = osp.join(test_output_dir, 'detections_withTracks.pkl')
    ran_once = False
    # all_det_files = [tracking_det_file, det_file]
    all_det_files = [tracking_det_file]
    for file_path in all_det_files:
        json_out_dir = osp.join(
            test_output_dir, osp.basename(file_path) + '_json/')
        if not osp.exists(file_path):
            continue
        ran_once = True
        logger.info('Evaluating {}'.format(file_path))
        _run_posetrack_eval(roidb, file_path, dataset, json_out_dir)
    if not ran_once:
        logger.warning('No detection files found from {}'.format(all_det_files))
