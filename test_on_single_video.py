#Script to run DetectandTrack Code on a single video which may or may not belong to any dataset.
 

import os
import os.path as osp
import sys

import numpy as np
import pickle
import cv2
import argparse
import shutil
import yaml
import glob
import time
from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace

from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from core.test_engine import initialize_model_from_cfg, empty_results, extend_results
from core.test import im_detect_all
from core.tracking_engine import _load_det_file, _write_det_file, _center_detections, _get_high_conf_boxes, _compute_matches

import utils.image as image_utils
import utils.video as video_utils
import utils.vis as vis_utils
import utils.subprocess as subprocess_utils
from utils.io import robust_pickle_dump
import utils.c2

try:
    cv2.ocl.setUseOpenCL(False)
except AttributeError:
    pass

MAX_TRACK_IDS = 999 
FIRST_TRACK_ID = 0


def parse_args():
    parser = argparse.ArgumentParser(description='Run DetectandTrack on a single video and visualize the results')
    parser.add_argument(
        '--cfg', '-c', dest='cfg_file', required=True,
        help='Config file to run')
    parser.add_argument(
        '--video', '-v', dest='video_path',
        help='Path to Video',
        required=True)
    parser.add_argument(
        '--output', '-o', dest='out_path',
        help='Path to Output')
    parser.add_argument(
        'opts', help='See lib/core/config.py for all options', default=None,
        nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()

def _read_video(args):
    vidcap = cv2.VideoCapture(args.video_path)
    success,image = vidcap.read()
    count = 1
    success = True
    temp_frame_folder = osp.join(args.out_path,args.vid_name + '_frames/')
    if os.path.exists(temp_frame_folder):
      shutil.rmtree(temp_frame_folder)
    os.makedirs(temp_frame_folder)  
    while success:
        cv2.imwrite(osp.join(temp_frame_folder,'%08d.jpg' % count), image)     # save frame as JPEG file      
        success,image = vidcap.read()
        count += 1
    return count-1

def _read_video_frames(out_path, vid_name, index):
    im = cv2.imread(osp.join(out_path,vid_name + '_frames','%08d.jpg'%(index+1)))
    assert im is not None
    return im

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
        show_box=True, show_class=False, linewidth = 1)
    if res is None:
        return im
    return res


def _generate_visualizations(entry, ix, all_boxes, all_keyps, all_tracks, thresh = 0.9):
    im = cv2.imread(entry)
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
    return pred

def _prune_bad_detections(dets, conf):
    """
    Keep only the boxes/poses that correspond to confidence > conf (float)
    """
    N = len(dets['all_boxes'][1])
    for i in range(N):
        boxes = dets['all_boxes'][1][i]
        poses = dets['all_keyps'][1][i]
        sel = np.where(np.squeeze(_get_high_conf_boxes(boxes, conf)))[0]
        boxes = boxes[sel]
        poses = [poses[j] for j in sel.tolist()]
        dets['all_boxes'][1][i] = boxes
        dets['all_keyps'][1][i] = poses
    return dets


def _compute_distance_matrix(
    prev_img_path, prev_boxes, prev_poses,
    cur_img_path, cur_boxes, cur_poses,
    cost_types, cost_weights, kps_names = None
):
    assert(len(cost_weights) == len(cost_types))
    all_Cs = []
    for cost_type, cost_weight in zip(cost_types, cost_weights):
        if cost_weight == 0:
            continue
        if cost_type == 'bbox-overlap':
            all_Cs.append((1 - _compute_pairwise_iou(prev_boxes, cur_boxes)))
        elif cost_type == 'cnn-cosdist':
            all_Cs.append(_compute_pairwise_deep_cosine_dist(
                prev_img_path, prev_boxes,
                curr_img_path, cur_boxes))
        elif cost_type == 'pose-pck':
            kps_names = cur_json_data['dataset'].person_cat_info['keypoints']
            all_Cs.append(_compute_pairwise_kpt_distance(
                prev_poses, cur_poses, kps_names))
        else:
            raise NotImplementedError('Unknown cost type {}'.format(cost_type))
        all_Cs[-1] *= cost_weight
    return np.sum(np.stack(all_Cs, axis=0), axis=0)


def _compute_tracks_video_lstm(frames, dets, lstm_model):
    nframes = len(frames)
    video_tracks = []
    next_track_id = FIRST_TRACK_ID
    # track_lstms contain track_id: <lstm_hidden_layer>
    track_lstms = {}
    for frame_id in range(nframes):
        frame_tracks = []
        # each element is (roidb entry, idx in the dets/original roidb)
        cur_boxes = dets['all_boxes'][1][frame_id]
        cur_poses = dets['all_keyps'][1][frame_id]
        cur_boxposes = lstm_track_utils.encode_box_poses(cur_boxes, cur_poses)
        # Compute LSTM next matches
        # Need to keep prev_track_ids to make sure of ordering of output
        prev_track_ids = video_tracks[frame_id - 1] if frame_id > 1 else []
        match_scores = lstm_track_utils.compute_matching_scores(
            track_lstms, prev_track_ids, cur_boxposes, lstm_model)
        if match_scores.size > 0:
            matches = _compute_matches(
                None, None, None, None, None, None, None, None,
                cfg.TRACKING.BIPARTITE_MATCHING_ALGO, C=(-match_scores))
        else:
            matches = -np.ones((cur_boxes.shape[0],))
        prev_tracks = video_tracks[frame_id - 1] if frame_id > 0 else None
        for m in matches:
            if m == -1:  # didn't match to any
                frame_tracks.append(next_track_id)
                next_track_id += 1
                if next_track_id >= MAX_TRACK_IDS:
                    logger.warning('Exceeded max track ids ({})'.format(
                        MAX_TRACK_IDS))
                    next_track_id %= MAX_TRACK_IDS
            else:
                frame_tracks.append(prev_tracks[m])
        # based on the matches, update the lstm hidden weights
        # Whatever don't get matched, start a new track ID. Whatever previous
        # track IDs don't get matched, have to be deleted.
        lstm_track_utils.update_lstms(
            track_lstms, prev_track_ids, frame_tracks, cur_boxposes, lstm_model)
        video_tracks.append(frame_tracks)
    return video_tracks

def _compute_tracks_video(frames, dets):
    nframes = len(frames)
    video_tracks = []
    next_track_id = FIRST_TRACK_ID
    for frame_id in range(nframes):
        frame_tracks = []
        # each element is (roidb entry, idx in the dets/original roidb
        cur_boxes = dets['all_boxes'][1][frame_id]
        cur_poses = dets['all_keyps'][1][frame_id]
        if (frame_id == 0):
            matches = -np.ones((cur_boxes.shape[0], ))
        else:
            cur_frame_data = frames[frame_id]
            prev_boxes = dets['all_boxes'][1][frame_id - 1]
            prev_poses = dets['all_keyps'][1][frame_id - 1]
                # 0-index to remove the other index to the dets structure
            prev_frame_data = frames[frame_id-1]
            matches = _compute_matches(
                prev_frame_data, cur_frame_data,
                prev_boxes, cur_boxes, prev_poses, cur_poses,
                cost_types=cfg.TRACKING.DISTANCE_METRICS,
                cost_weights=cfg.TRACKING.DISTANCE_METRIC_WTS,
                bipart_match_algo=cfg.TRACKING.BIPARTITE_MATCHING_ALGO)
        prev_tracks = video_tracks[frame_id - 1] if frame_id > 0 else None
        for m in matches:
            if m == -1:  # didn't match to any
                frame_tracks.append(next_track_id)
                next_track_id += 1
                if next_track_id >= MAX_TRACK_IDS:
                    next_track_id %= MAX_TRACK_IDS
            else:
                frame_tracks.append(prev_tracks[m])
        video_tracks.append(frame_tracks)
    return video_tracks

def compute_matches_tracks(frames, dets, lstm_model):
    # Consider all consecutive frames, and match the boxes
    num_images = len(frames)
    all_tracks = [[]] * num_images
    if cfg.TRACKING.LSTM_TEST.LSTM_TRACKING_ON:
        tracks = _compute_tracks_video_lstm(frames, dets, lstm_model)
    else:
        tracks = _compute_tracks_video(frames, dets)
    if cfg.TRACKING.FLOW_SMOOTHING_ON:
        tracks = _smooth_pose_video(
                frames, dets, tracks)
        # resort and assign
    for i in range(num_images):
        all_tracks[i] = tracks[i]
    dets['all_tracks'] = [[], all_tracks]
    return dets

def main(name_scope, gpu_dev, num_images, args):
    
    model = initialize_model_from_cfg()
    num_classes = cfg.MODEL.NUM_CLASSES
    all_boxes, all_segms, all_keyps = empty_results(num_classes, num_images)
    for i in range(num_images):
        print('Processing Detection for Image %d'%(i+1))
        im_ = _read_video_frames(args.out_path, args.vid_name, i) 
        im_ = np.expand_dims(im_, 0)
        with core.NameScope(name_scope):
            with core.DeviceScope(gpu_dev):
                cls_boxes_i, cls_segms_i, cls_keyps_i = im_detect_all(
                    model, im_, None) #TODO: Parallelize detection

	    extend_results(i, all_boxes, cls_boxes_i)
        if cls_segms_i is not None:
            extend_results(i, all_segms, cls_segms_i)
        if cls_keyps_i is not None:
            extend_results(i, all_keyps, cls_keyps_i)

        im_name = '%08d.jpg'%(i+1)
        im_ = im_.squeeze()
    cfg_yaml = yaml.dump(cfg)
    
    det_name = args.vid_name + '_detections.pkl'
    det_file = osp.join(args.out_path, det_name)
    robust_pickle_dump(
        dict(all_boxes=all_boxes,
             all_segms=all_segms,
             all_keyps=all_keyps,
             cfg=cfg_yaml),
        det_file)

    frames = sorted(glob.glob(osp.join(args.out_path,args.vid_name + '_frames','*.jpg')))
    out_detrack_file = osp.join(args.out_path, args.vid_name + '_detections_withTracks.pkl')

    # Debug configurations
    if cfg.TRACKING.DEBUG.UPPER_BOUND_2_GT_KPS:  # if this is true
        cfg.TRACKING.DEBUG.UPPER_BOUND = True  # This must be set true

    # Set include_gt True when using the roidb to evalute directly. Not doing
    # that currently
    dets = _load_det_file(det_file)
    if cfg.TRACKING.KEEP_CENTER_DETS_ONLY:
        _center_detections(dets)

    conf = cfg.TRACKING.CONF_FILTER_INITIAL_DETS
    dets = _prune_bad_detections(dets, conf)
    if cfg.TRACKING.LSTM_TEST.LSTM_TRACKING_ON:
        # Needs torch, only importing if we need to run LSTM tracking
        from lstm.lstm_track import lstm_track_utils
        lstm_model = lstm_track_utils.init_lstm_model(
            cfg.TRACKING.LSTM_TEST.LSTM_WEIGHTS)
        lstm_model.cuda()
    else:
        lstm_model = None
    tic_track = time.time()
    dets_withTracks = compute_matches_tracks(frames, dets, lstm_model)
    toc_track = time.time()
    _write_det_file(dets_withTracks, out_detrack_file)
    
    for i in range(num_images):
        vis_im = _generate_visualizations(frames[i], i, dets['all_boxes'], dets['all_keyps'], dets['all_tracks'])
        cv2.imwrite(osp.join(args.out_path, args.vid_name + '_vis','%08d.jpg'%(i+1)),vis_im)

if __name__=='__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    args = parse_args()
    if args.out_path == None:
        args.out_path = args.video_path
    args.vid_name = args.video_path.split('/')[-1].split('.')[0]
    utils.c2.import_custom_ops()
    utils.c2.import_detectron_ops()
    utils.c2.import_contrib_ops()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.opts is not None:
        cfg_from_list(args.opts)
    assert_and_infer_cfg()
    if osp.exists(osp.join(args.out_path,args.vid_name + '_vis')):
        shutil.rmtree(osp.join(args.out_path, args.vid_name + '_vis'))
    os.makedirs(osp.join(args.out_path,args.vid_name+ '_vis'))
    num_images = _read_video(args)
    gpu_dev = core.DeviceOption(caffe2_pb2.CUDA, cfg.ROOT_GPU_ID)
    name_scope = 'gpu_{}'.format(cfg.ROOT_GPU_ID)
    main(name_scope, gpu_dev, num_images, args)
