##############################################################
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################

"""Compute tracks in a detection file over a video using Hungarian algo."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals


import sys
import cPickle as pkl
import json
import os.path as osp
import numpy as np
import scipy.optimize
import scipy.spatial
from tqdm import tqdm
import logging
import cv2
import multiprocessing as mp
from contextlib import closing
import copy

from datasets.json_dataset import JsonDataset
from core.config import cfg
import utils.boxes as box_utils
import utils.general as gen_utils
import utils.keypoints as kps_utils
import utils.image as img_utils
import utils.vis as vis_utils
from core.mpii_eval_engine import run_mpii_eval

FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)
np.set_printoptions(suppress=True)

# This is weird.. do something about this..
MAX_TRACK_IDS = 999  # earlier code used to allow only 100
FIRST_TRACK_ID = 0


def _load_det_file(det_fpath):
    with open(det_fpath, 'rb') as fin:
        return pkl.load(fin)


def _write_det_file(dets, det_fpath):
    with open(det_fpath, 'wb') as fout:
        pkl.dump(dets, fout, pkl.HIGHEST_PROTOCOL)


def _load_json_file(json_fpath):
    with open(json_fpath, 'rb') as fin:
        return json.load(fin)


def _is_same_video(json1, json2):
    return (osp.dirname(img_utils.get_image_path(json1)) ==
            osp.dirname(img_utils.get_image_path(json2)))


def _get_boxes(dets, img_id):
    return dets['all_boxes'][1][img_id]


def _get_poses(dets, img_id):
    return dets['all_keyps'][1][img_id]


def _set_poses(poses, dets, img_id):
    dets['all_keyps'][1][img_id] = poses


def _set_boxes(boxes, dets, img_id):
    dets['all_boxes'][1][img_id] = boxes


def _center_boxes(boxes):
    if len(boxes) == 0:
        return boxes
    assert (boxes.shape[-1] - 1) % 4 == 0, 'Must contain scores in last col.'
    time_dim = (boxes.shape[-1] - 1) // 4
    center = time_dim // 2
    res = boxes[:, np.array(range(center * 4, (center + 1) * 4) + [-1])]
    return res


def _center_poses(poses):
    if len(poses) == 0:
        return poses
    time_dim = poses[0].shape[-1] // cfg.KRCNN.NUM_KEYPOINTS
    center = time_dim // 2
    res = [el[..., center * cfg.KRCNN.NUM_KEYPOINTS:
              (center + 1) * cfg.KRCNN.NUM_KEYPOINTS] for el in poses]
    return res


def _compute_pairwise_iou(a, b):
    """
    a, b (np.ndarray) of shape Nx4T and Mx4T.
    The output is NxM, for each combination of boxes.
    """
    return box_utils.bbox_overlaps(a, b)


def _compute_pairwise_kpt_distance(a, b, kpt_names):
    """
    Args:
        a, b (poses): Two sets of poses to match
        Each "poses" is represented as a list of 3x17 or 4x17 np.ndarray
    This tries to recreate the assignGT function from the evaluation code_dir
    https://github.com/leonid-pishchulin/poseval/blob/954d8d84f459e942a185f835fc2a0fbdee5ce354/py/eval_helpers.py#L423  # noQA
    Main points:
        prToGT is the prediction_to_gt output that I want to recreate
        Essentially it represents a form of PCK metric
    """
    res = np.zeros((len(a), len(b)))
    for i in range(len(a)):
        for j in range(len(b)):
            res[i, j] = kps_utils.pck_distance(a[i], b[j], kpt_names)
    return res


def _compute_deep_features(imname, boxes):
    # Not doing an import at the root level as I'm no longer compiling with
    # torch. Was too slow in compilation and am not using it anyway. If I
    # end up using later, uncomment the torch dependencies in TARGETS and move
    # this to the top of the file.
    import utils.pytorch_cnn_features as cnn_utils
    I = cv2.imread(imname)
    if I is None:
        raise ValueError('Image not found {}'.format(imname))
    all_feats = []
    for box in boxes:
        patch = I[int(box[1]): int(box[3]), int(box[0]): int(box[2]), :]
        all_feats.append(cnn_utils.extract_features(
            patch, layers=(cfg.TRACKING.CNN_MATCHING_LAYER,)))
    return np.stack(all_feats) if len(all_feats) > 0 else np.zeros((0, ))


def _compute_pairwise_deep_cosine_dist(a_imname, a, b_imname, b):
    f1 = _compute_deep_features(a_imname, a)
    f2 = _compute_deep_features(b_imname, b)
    if f1.size * f2.size == 0:  # either were 0
        return np.zeros((f1.shape[0], f2.shape[0]))
    return scipy.spatial.distance.cdist(
        f1.reshape((f1.shape[0], -1)), f2.reshape((f2.shape[0], -1)), 'cosine')


def _compute_distance_matrix(
    prev_json_data, prev_boxes, prev_poses,
    cur_json_data, cur_boxes, cur_poses,
    cost_types, cost_weights,
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
                img_utils.get_image_path(prev_json_data), prev_boxes,
                img_utils.get_image_path(cur_json_data), cur_boxes))
        elif cost_type == 'pose-pck':
            kps_names = cur_json_data['dataset'].person_cat_info['keypoints']
            all_Cs.append(_compute_pairwise_kpt_distance(
                prev_poses, cur_poses, kps_names))
        else:
            raise NotImplementedError('Unknown cost type {}'.format(cost_type))
        all_Cs[-1] *= cost_weight
    return np.sum(np.stack(all_Cs, axis=0), axis=0)


def bipartite_matching_greedy(C):
    """
    Computes the bipartite matching between the rows and columns, given the
    cost matrix, C.
    """
    C = C.copy()  # to avoid affecting the original matrix
    prev_ids = []
    cur_ids = []
    row_ids = np.arange(C.shape[0])
    col_ids = np.arange(C.shape[1])
    while C.size > 0:
        # Find the lowest cost element
        i, j = np.unravel_index(C.argmin(), C.shape)
        # Add to results and remove from the cost matrix
        row_id = row_ids[i]
        col_id = col_ids[j]
        prev_ids.append(row_id)
        cur_ids.append(col_id)
        C = np.delete(C, i, 0)
        C = np.delete(C, j, 1)
        row_ids = np.delete(row_ids, i, 0)
        col_ids = np.delete(col_ids, j, 0)
    return prev_ids, cur_ids


def _compute_matches(prev_frame_data, cur_frame_data, prev_boxes, cur_boxes,
                     prev_poses, cur_poses,
                     cost_types, cost_weights,
                     bipart_match_algo, C=None):
    """
    C (cost matrix): num_prev_boxes x num_current_boxes
    Optionally input the cost matrix, in which case you can input dummy values
    for the boxes and poses
    Returns:
        matches: A 1D np.ndarray with as many elements as boxes in current
        frame (cur_boxes). For each, there is an integer to index the previous
        frame box that it matches to, or -1 if it doesnot match to any previous
        box.
    """
    # matches structure keeps track of which of the current boxes matches to
    # which box in the previous frame. If any idx remains -1, it will be set
    # as a new track.
    if C is None:
        nboxes = cur_boxes.shape[0]
        matches = -np.ones((nboxes,), dtype=np.int32)
        C = _compute_distance_matrix(
            prev_frame_data, prev_boxes, prev_poses,
            cur_frame_data, cur_boxes, cur_poses,
            cost_types=cost_types,
            cost_weights=cost_weights)
    else:
        matches = -np.ones((C.shape[1],), dtype=np.int32)
    if bipart_match_algo == 'hungarian':
        prev_inds, next_inds = scipy.optimize.linear_sum_assignment(C)
    elif bipart_match_algo == 'greedy':
        prev_inds, next_inds = bipartite_matching_greedy(C)
    else:
        raise NotImplementedError('Unknown matching algo: {}'.format(
            bipart_match_algo))
    assert(len(prev_inds) == len(next_inds))
    for i in range(len(prev_inds)):
        matches[next_inds[i]] = prev_inds[i]
    return matches


def _known_shot_change(video_json_data, frame_id):
    if not cfg.TRACKING.DEBUG.UPPER_BOUND_3_SHOTS:
        # Only use it if in debugging mode, to get a upper bound
        return False
    if frame_id == 0:
        return True
    # read the CSV
    import csv
    D = {}
    with open('/path/to/shot_boundaries_val.csv', 'r') as fin:
        reader = csv.reader(fin)
        for row in reader:
            D[row[0]] = [int(el) for el in row[1].strip().split(',') if len(el) > 0]
    vname = osp.dirname(img_utils.get_image_path(video_json_data[frame_id][0]))
    vname = osp.basename(osp.dirname(vname)) + '/' + osp.basename(vname)
    assert(vname in D)
    # frame_id + 1 since the frame_id are 0 indexed and I labeled as per 1-index
    if frame_id + 1 in D[vname]:
        return True
    else:
        return False


def _compute_tracks_video(video_json_data, dets):
    nframes = len(video_json_data)
    video_tracks = []
    next_track_id = FIRST_TRACK_ID
    for frame_id in range(nframes):
        frame_tracks = []
        # each element is (roidb entry, idx in the dets/original roidb)
        frame_data, det_id = video_json_data[frame_id]
        cur_boxes = _get_boxes(dets, det_id)
        cur_poses = _get_poses(dets, det_id)
        if (frame_id == 0 or _known_shot_change(video_json_data, frame_id)) \
                and not cfg.TRACKING.DEBUG.UPPER_BOUND:
            matches = -np.ones((cur_boxes.shape[0], ))
        else:
            cur_frame_data = frame_data
            if cfg.TRACKING.DEBUG.UPPER_BOUND:
                prev_boxes = frame_data['boxes']
                prev_poses = [el for el in frame_data['gt_keypoints']]
                prev_frame_data = {'image': img_utils.get_image_path(cur_frame_data)}
            else:
                prev_boxes = _get_boxes(dets, video_json_data[frame_id - 1][1])
                prev_poses = _get_poses(dets, video_json_data[frame_id - 1][1])
                # 0-index to remove the other index to the dets structure
                prev_frame_data = video_json_data[frame_id - 1][0]
            matches = _compute_matches(
                prev_frame_data, cur_frame_data,
                prev_boxes, cur_boxes, prev_poses, cur_poses,
                cost_types=cfg.TRACKING.DISTANCE_METRICS,
                cost_weights=cfg.TRACKING.DISTANCE_METRIC_WTS,
                bipart_match_algo=cfg.TRACKING.BIPARTITE_MATCHING_ALGO)
        if cfg.TRACKING.DEBUG.UPPER_BOUND:
            prev_tracks = frame_data['tracks'].reshape((-1)).tolist()
            matched = np.where(np.array(matches) != -1)[0]
            # Remove things unmatched
            matches = matches[matched]
            new_boxes = _get_boxes(dets, det_id)[matched]
            # This doesn't help, but made the pose score go low
            # new_boxes[:, -1] = 1.0  # make the detections 100% confidence
            new_poses = [_get_poses(dets, det_id)[el] for el in matched]
            if cfg.TRACKING.DEBUG.UPPER_BOUND_2_GT_KPS:
                # Set the points to be GT points
                new_boxes[:, :4] = frame_data['boxes'][matches]
                for match_id in range(matches.shape[0]):
                    if cfg.TRACKING.DEBUG.UPPER_BOUND_2_GT_KPS_ONLY_CONF:
                        dims_to_replace = np.array(2)
                    else:
                        dims_to_replace = np.arange(3)
                    new_poses[match_id][dims_to_replace, :] = frame_data[
                        'gt_keypoints'][matches[match_id]][dims_to_replace, :]
            _set_boxes(new_boxes, dets, det_id)
            _set_poses(new_poses, dets, det_id)
        else:
            prev_tracks = video_tracks[frame_id - 1] if frame_id > 0 else None
        if cfg.TRACKING.DEBUG.UPPER_BOUND_5_GT_KPS_ONLY:
            gt_prev_boxes = frame_data['boxes']
            gt_prev_poses = [el for el in frame_data['gt_keypoints']]
            gt_prev_frame_data = {'image': img_utils.get_image_path(frame_data)}
            matches_gt = _compute_matches(
                gt_prev_frame_data, frame_data,
                gt_prev_boxes, cur_boxes, gt_prev_poses, cur_poses,
                cost_types=('bbox-overlap', ),
                cost_weights=(1.0, ), bipart_match_algo='hungarian')
            # replace the predicted poses
            for match_id in range(matches_gt.shape[0]):
                if matches_gt[match_id] == -1:
                    continue
                cur_poses[match_id][:3, :] = gt_prev_poses[matches_gt[match_id]][:3, :]
        for m in matches:
            if m == -1:  # didn't match to any
                frame_tracks.append(next_track_id)
                next_track_id += 1
                if next_track_id >= MAX_TRACK_IDS:
                    logger.warning('Exceeded max track ids ({}) for {}'.format(
                        MAX_TRACK_IDS, frame_data['image']))
                    next_track_id %= MAX_TRACK_IDS
            else:
                frame_tracks.append(prev_tracks[m])
        video_tracks.append(frame_tracks)
    return video_tracks


def _compute_tracks_video_lstm(video_json_data, dets, lstm_model):
    nframes = len(video_json_data)
    video_tracks = []
    next_track_id = FIRST_TRACK_ID
    # track_lstms contain track_id: <lstm_hidden_layer>
    track_lstms = {}
    for frame_id in range(nframes):
        frame_tracks = []
        # each element is (roidb entry, idx in the dets/original roidb)
        frame_data, det_id = video_json_data[frame_id]
        cur_boxes = _get_boxes(dets, det_id)
        cur_poses = _get_poses(dets, det_id)
        cur_boxposes = lstm_track_utils.encode_box_poses(cur_boxes, cur_poses)
        # Compute LSTM next matches
        # Need to keep prev_track_ids to make sure of ordering of output
        prev_track_ids = video_tracks[frame_id - 1] if frame_id > 1 else []
        match_scores = lstm_track_utils.compute_matching_scores(
            track_lstms, prev_track_ids, cur_boxposes, lstm_model)
        # Trying to see what random scores do.
        # match_scores = np.random.random((len(prev_track_ids), len(cur_boxposes)))
        # Compute the matching. -match_scores as I need to pass in the distance
        # values
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
                    logger.warning('Exceeded max track ids ({}) for {}'.format(
                        MAX_TRACK_IDS, frame_data['image']))
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


def _shift_using_flow(poses, flow):
    # too large flow => must be a shot boundary. So don't transfer poses over
    # shot boundaries
    if flow is None:
        return []
    if np.abs(flow).mean() > cfg.TRACKING.FLOW_SMOOTHING.FLOW_SHOT_BOUNDARY_TH:
        return []
    res = []
    for pose in poses:
        res_pose = pose.copy()
        x = np.round(pose[0].copy())
        y = np.round(pose[1].copy())
        valid = np.logical_and(
            np.logical_and(x >= 0, x < flow.shape[1]),
            np.logical_and(y >= 0, y < flow.shape[0]))
        x[np.logical_not(valid)] = 0
        y[np.logical_not(valid)] = 0
        delta_x = flow[y.astype('int'), x.astype('int'), 0]
        delta_y = flow[y.astype('int'), x.astype('int'), 1]
        delta_x[np.logical_not(valid)] = 0
        delta_y[np.logical_not(valid)] = 0
        # Since I wasn't able to move the points for which a valid flow vector
        # did not exist, set confidence to 0
        res_pose[2, np.logical_not(valid)] = 0
        res_pose[0, :] += delta_x
        res_pose[1, :] += delta_y
        res.append(res_pose)
    return res


def _shift_poses(poses_boxes_tracks, delta, flows):
    nframes = len(poses_boxes_tracks)
    res_poses = [([], [], [])] * nframes
    for frame_id in range(nframes):
        target_frame_id = frame_id + delta
        if target_frame_id >= 0 and target_frame_id < nframes:
            if len(poses_boxes_tracks[frame_id][0]) != 0:
                try:
                    shifted_pose = _shift_using_flow(
                        poses_boxes_tracks[frame_id][0], flows[frame_id])
                except Exception:
                    shifted_pose = []
                if len(shifted_pose) != 0:
                    shifted_boxes = poses_boxes_tracks[frame_id][1]
                    shifted_tracks = poses_boxes_tracks[frame_id][2]
                    res_poses[target_frame_id] = (
                        shifted_pose,
                        shifted_boxes,
                        shifted_tracks,
                    )
    return res_poses


def _weighted_avg_poses(poses, shifts):
    final = np.full((4, 17), -np.inf, dtype=np.float32)
    for pose, shift in zip(poses, shifts):
        wt = pow(1.2, -abs(shift))
        if len(pose) > 0:
            good_pts = pose[2, :] * wt > final[2, :]
            final[:, good_pts] = pose[:, good_pts]
            final[2, good_pts] = final[2, good_pts] * wt
    return final


def _combine_boxes(boxes):
    # remove empty boxes
    boxes = [el for el in boxes if len(el) > 0]
    assert(len(boxes) > 0)  # at least one of the frames must have the track
    # Max pool over time
    res = np.zeros_like(boxes[0])
    boxes = np.stack(boxes)
    score = np.max(boxes[:, -1])  # score
    boxes = boxes[:, :-1]  # remove the score now.
    res[0::4] = np.min(boxes[:, 0::4], axis=0)
    res[1::4] = np.min(boxes[:, 1::4], axis=0)
    res[2::4] = np.max(boxes[:, 2::4], axis=0)
    res[3::4] = np.max(boxes[:, 3::4], axis=0)
    res[-1] = score
    return res


def _combine_shifted_poses(video_json_data, nframes, shifted_poses_boxes_tracks):
    res_all_poses = []
    res_all_boxes = []
    res_all_tracks = []
    shifts = sorted(shifted_poses_boxes_tracks.keys())
    for frame_id in range(nframes):
        res_frame_poses = []
        res_frame_boxes = []
        res_frame_tracks = []
        tracks = set()
        if cfg.TRACKING.FLOW_SMOOTHING.EXTEND_TRACKS:
            for shift in shifts:
                _, _, frame_tracks = shifted_poses_boxes_tracks[shift][frame_id]
                tracks |= set(frame_tracks)
            tracks = list(tracks)
        else:
            _, _, tracks = shifted_poses_boxes_tracks[0][frame_id]
        # even if 0 boxes, this will have (0, 4t+1) shape
        box_size = shifted_poses_boxes_tracks[0][frame_id][1].shape[-1]
        for track_id in tracks:
            # collect all poses in context frames with same track_id
            poses_over_time = []
            boxes_over_time = []
            for shift in shifts:
                shifted_poses, shifted_boxes, shifted_tracks = \
                    shifted_poses_boxes_tracks[shift][frame_id]
                if track_id in shifted_tracks:
                    poses_over_time.append(shifted_poses[shifted_tracks.index(
                        track_id)])
                    boxes_over_time.append(shifted_boxes[shifted_tracks.index(
                        track_id)])
                else:
                    poses_over_time.append([])
                    boxes_over_time.append([])
            pose = _weighted_avg_poses(poses_over_time, shifts)
            box = _combine_boxes(boxes_over_time)
            if cfg.TRACKING.DEBUG.FLOW_SMOOTHING_COMBINE and \
                    np.random.random() < 0.005:
                # Stop only for a subset of cases
                vis = video_json_data[frame_id][0]
                for box_i, pose_i in zip(boxes_over_time, poses_over_time):
                    if len(box_i) == 0:
                        continue
                    vis = vis_utils.vis_predictions(vis, np.array([box_i]), [pose_i])
                cv2.imwrite('/tmp/temp.jpg', vis[0])
                vis = vis_utils.vis_predictions(
                    video_json_data[frame_id][0], np.array([box]), [pose])
                cv2.imwrite('/tmp/temp2.jpg', vis[0])
                vis = vis_utils.vis_predictions(
                    video_json_data[frame_id][0],
                    np.array([boxes_over_time[0]]),
                    [poses_over_time[0]])
                cv2.imwrite('/tmp/temp3.jpg', vis[0])
                import pdb
                pdb.set_trace()
            res_frame_poses.append(pose)
            res_frame_boxes.append(box)
            res_frame_tracks.append(track_id)
        res_all_poses.append(res_frame_poses)
        if len(res_frame_boxes) > 0:
            res_all_boxes.append(np.stack(res_frame_boxes))
        else:
            res_all_boxes.append(np.zeros((0, box_size)))
        res_all_tracks.append(res_frame_tracks)
    return res_all_poses, res_all_boxes, res_all_tracks


def run_farneback(frames):
    try:
        return cv2.calcOpticalFlowFarneback(
            frames[0], frames[1],
            # options, defaults
            None,  # output
            0.5,  # pyr_scale, 0.5
            10,  # levels, 3
            min(frames[0].shape[:2]) // 5,  # winsize, 15
            10,  # iterations, 3
            7,  # poly_n, 5
            1.5,  # poly_sigma, 1.2
            cv2.OPTFLOW_FARNEBACK_GAUSSIAN,  # flags, 0
        )
    except cv2.error:
        return None


def _compute_neg_flow(flow):
    if flow is None:
        return None
    return -flow


def compute_optical_flow(video_json_data):
    frames = [img_utils.read_image_video(el, key_frame_only=True)[0]
              for el, _ in video_json_data]
    if len(frames) == 0:
        return
    frames = [cv2.cvtColor(el.astype('uint8'), cv2.COLOR_BGR2GRAY) for el in frames]
    flows = []
    neg_flows = []
    all_pairs = [(frames[i], frames[i + 1]) for i in range(len(frames) - 1)]
    with closing(mp.Pool(32)) as pool:
        # https://stackoverflow.com/a/25968716/1492614
        all_pairs_flow = list(tqdm(pool.imap(run_farneback, all_pairs),
                                   total=len(all_pairs),
                                   desc='Computing flow',
                                   leave=False))
        pool.terminate()
    # No negative flow defined for the fist frame
    neg_flows.append(np.zeros((frames[0].shape[0], frames[0].shape[1], 2)))
    for frame_id in range(len(all_pairs_flow)):
        flow = all_pairs_flow[frame_id]
        flows.append(flow)
        neg_flows.append(_compute_neg_flow(flow))
    # no flow defined for last frame
    flows.append(np.zeros((frames[0].shape[0], frames[0].shape[1], 2)))
    return flows, neg_flows


def _smooth_pose_video(video_json_data, dets, tracks):
    """
    Smooth the pose over frames.
    Args:
        ncontext (int): Number of frames to consider forward and backward.
        So, will look at total of (2 * ncontext + 1) frames
    """
    ncontext = cfg.TRACKING.FLOW_SMOOTHING.N_CONTEXT_FRAMES
    if ncontext > 0:
        flows, neg_flows = compute_optical_flow(video_json_data)
    else:
        flows, neg_flows = [[], []]
    assert(ncontext >= 0)
    nframes = len(video_json_data)
    # extract all poses
    poses = []
    boxes = []
    for frame_id in range(nframes):
        frame_data, det_id = video_json_data[frame_id]
        poses.append(_get_poses(dets, det_id))
        boxes.append(_get_boxes(dets, det_id))
    # compute the shifted poses
    shifted_poses_boxes_tracks = {}
    shifted_poses_boxes_tracks[0] = copy.deepcopy(zip(poses, boxes, tracks))
    # move forward in time
    for i in range(1, ncontext + 1):
        shifted_poses_boxes_tracks[i] = _shift_poses(
            shifted_poses_boxes_tracks[i - 1], +1, flows)
        # move backward in time
    for i in range(-ncontext, 0)[::-1]:
        shifted_poses_boxes_tracks[i] = _shift_poses(
            shifted_poses_boxes_tracks[i + 1], -1, neg_flows)
    # combine all shifted poses
    new_poses, new_boxes, new_tracks = _combine_shifted_poses(
        video_json_data, nframes, shifted_poses_boxes_tracks)
    if ncontext == 0:
        # make sure get the same output as input
        for frame_id in range(nframes):
            assert(np.all(boxes[frame_id] == new_boxes[frame_id]))
    for frame_id in range(nframes):
        frame_data, det_id = video_json_data[frame_id]
        _set_poses(new_poses[frame_id], dets, det_id)
        _set_boxes(new_boxes[frame_id], dets, det_id)
    return new_tracks


def _summarize_track_stats(all_tracks, json_data):
    all_lengths = []
    assert(len(all_tracks) == len(json_data))
    for track_id in range(FIRST_TRACK_ID, MAX_TRACK_IDS + 1):
        # compute the lengths of all tracks of this track ID
        lengths = []
        cur_len = 0
        for i in range(len(all_tracks)):
            tracks = all_tracks[i]
            if track_id in tracks and (
                    i == 0 or _is_same_video(json_data[i - 1], json_data[i])):
                cur_len += 1
            else:
                if cur_len > 0:
                    lengths.append(cur_len)
                    cur_len = 0 if track_id not in tracks else 1
        all_lengths += lengths
    print('Track length (min/avg/max): {} {} {}'.format(
        np.min(all_lengths),
        np.mean(all_lengths),
        np.max(all_lengths)))


def compute_matches_tracks(json_data, dets, lstm_model):
    # Consider all consecutive frames, and match the boxes
    num_imgs = len(json_data)
    all_tracks = [[]] * len(json_data)
    # First split the images into videos
    all_video_roidb = []
    video_entries = []
    for img_id in range(num_imgs):
        if img_id == 0 or _is_same_video(
                json_data[img_id - 1], json_data[img_id]):
            video_entries.append((json_data[img_id], img_id))
        else:
            all_video_roidb.append(sorted(
                video_entries, key=lambda x: img_utils.get_image_path(x[0])))
            video_entries = [(json_data[img_id], img_id)]
    if len(video_entries) > 0:
        all_video_roidb.append(video_entries)
    # Make sure I got everything
    assert(len(json_data) == len(gen_utils.flatten_list(all_video_roidb)))
    logger.info('Computing tracks for {} videos.'.format(len(all_video_roidb)))
    for vid_id in tqdm(range(len(all_video_roidb)), desc='Tracks compute'):
        if cfg.TRACKING.LSTM_TEST.LSTM_TRACKING_ON:
            tracks = _compute_tracks_video_lstm(
                all_video_roidb[vid_id], dets, lstm_model)
        else:
            tracks = _compute_tracks_video(all_video_roidb[vid_id], dets)
        if cfg.TRACKING.FLOW_SMOOTHING_ON:
            tracks = _smooth_pose_video(
                all_video_roidb[vid_id], dets, tracks)
        # resort and assign
        for i, (_, det_id) in enumerate(all_video_roidb[vid_id]):
            all_tracks[det_id] = tracks[i]
            if cfg.TRACKING.DEBUG.DUMMY_TRACKS:
                # Replace with random track IDs
                all_tracks[det_id] = [
                    np.random.randint(FIRST_TRACK_ID, MAX_TRACK_IDS + 1) for
                    _ in tracks[i]]
    dets['all_tracks'] = [[], all_tracks]
    _summarize_track_stats(all_tracks, json_data)
    return dets


def _get_high_conf_boxes(boxes, conf):
    return boxes[:, -1] >= conf


def _get_big_size_boxes(boxes):
    area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return area >= 50


def _get_big_inside_image_boxes(boxes, im_data):
    ht = im_data['height']
    wd = im_data['width']
    nboxes = boxes.shape[0]
    boxes[:, 0] = np.maximum(boxes[:, 0], 0 * np.ones((nboxes, )))
    boxes[:, 1] = np.maximum(boxes[:, 1], 0 * np.ones((nboxes, )))
    boxes[:, 2] = np.minimum(boxes[:, 2], wd * np.ones((nboxes, )))
    boxes[:, 3] = np.minimum(boxes[:, 3], ht * np.ones((nboxes, )))
    return _get_big_size_boxes(boxes)


def _prune_bad_detections(dets, json_data, conf):
    """
    Keep only the boxes/poses that correspond to confidence > conf (float),
    and are big enough inside the image.
    """
    N = len(dets['all_boxes'][1])
    for i in range(N):
        boxes = dets['all_boxes'][1][i]
        poses = dets['all_keyps'][1][i]
        sel = np.where(np.logical_and.reduce((
            _get_high_conf_boxes(boxes, conf),
            _get_big_inside_image_boxes(boxes, json_data[i]),
        )))[0]
        boxes = boxes[sel]
        poses = [poses[j] for j in sel.tolist()]
        dets['all_boxes'][1][i] = boxes
        dets['all_keyps'][1][i] = poses
    return dets


def _center_detections(dets):
    N = len(dets['all_boxes'][1])
    for img_id in range(N):
        _set_boxes(_center_boxes(_get_boxes(dets, img_id)), dets, img_id)
        _set_poses(_center_poses(_get_poses(dets, img_id)), dets, img_id)


def run_posetrack_tracking(test_output_dir, json_data):
    if len(cfg.TRACKING.DETECTIONS_FILE):
        det_file = cfg.TRACKING.DETECTIONS_FILE
    else:
        det_file = osp.join(test_output_dir, 'detections.pkl')
    out_det_file = osp.join(test_output_dir, 'detections_withTracks.pkl')
    if not osp.exists(det_file):
        raise ValueError('Output file not found {}'.format(det_file))
    else:
        logger.info('Tracking over {}'.format(det_file))

    # Debug configurations
    if cfg.TRACKING.DEBUG.UPPER_BOUND_2_GT_KPS:  # if this is true
        cfg.TRACKING.DEBUG.UPPER_BOUND = True  # This must be set true

    # Set include_gt True when using the roidb to evalute directly. Not doing
    # that currently
    dets = _load_det_file(det_file)
    if cfg.TRACKING.KEEP_CENTER_DETS_ONLY:
        _center_detections(dets)
    assert(len(json_data) == len(dets['all_boxes'][1]))
    assert(len(json_data) == len(dets['all_keyps'][1]))
    conf = cfg.TRACKING.CONF_FILTER_INITIAL_DETS
    logger.info('Pruning detections with less than {} confidence'.format(conf))
    dets = _prune_bad_detections(dets, json_data, conf)
    if cfg.TRACKING.LSTM_TEST.LSTM_TRACKING_ON:
        # Needs torch, only importing if we need to run LSTM tracking
        from lstm.lstm_track import lstm_track_utils
        lstm_model = lstm_track_utils.init_lstm_model(
            cfg.TRACKING.LSTM_TEST.LSTM_WEIGHTS)
        lstm_model.cuda()
    else:
        lstm_model = None
    dets_withTracks = compute_matches_tracks(json_data, dets, lstm_model)
    _write_det_file(dets_withTracks, out_det_file)
    dataset = JsonDataset(cfg.TEST.DATASET)
    if dataset.name.startswith('posetrack'):
        run_mpii_eval(test_output_dir, json_data, dataset)
