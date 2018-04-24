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

import cv2
import os
import numpy as np
import scipy.sparse
from tqdm import tqdm
import logging
import sys
import math

from core.config import cfg
import utils.general as gen_utils

# OpenCL is enabled by default in OpenCV3 and it is not thread-safe leading
# to huge GPU memory allocations. See https://fburl.com/9d7tvusd
try:
  cv2.ocl.setUseOpenCL(False)
except AttributeError:
  pass

FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def get_video_info(roidb):
    """
    For each entry in roidb, returns a dictionary with
        (video_name, key_frame, flipped or not)
    """
    video_frames = {}
    for i, entry in enumerate(roidb):
        if entry['dataset'].frames_from_video:
            # For a video dataset like Kinetics
            video_name = entry['image']
            key_frame = entry['frame_id']
        else:
            video_name = os.path.dirname(entry['image'])
            key_frame = int(os.path.splitext(os.path.basename(
                entry['image']))[0])
        video_frames[i] = (video_name, key_frame, entry['flipped'])
    return video_frames


def _center_crop_list(l, sz):
    assert(len(l) >= sz)
    start_pt = (len(l) // 2) - (sz // 2)
    res = l[start_pt: (start_pt + sz)]
    assert len(res) == sz, 'Make sure got right number of elts'
    assert res[len(res) // 2] == l[len(l) // 2], 'Make sure got the center correct'
    return res


def _combine_clips(entry):
    """ entry[clip_ids] contains the various frames. Combine them all
    into the main entry to construct tubes etc.
    """
    new_entry = {}
    new_entry['image'] = [clip['image'] for clip in entry['clip_ids']]
    # take a subset of the clip ids now for the remaining stuff
    assert(cfg.VIDEO.NUM_FRAMES_MID <= cfg.VIDEO.NUM_FRAMES)
    entry['clip_ids'] = _center_crop_list(entry['clip_ids'], cfg.VIDEO.NUM_FRAMES_MID)
    copy_fields = [
        'dataset',
        'has_visible_keypoints',
        'id',
        'nframes',
        'width',
        'head_boxes',
        'is_labeled',
        'frame_id',
        'height',
        'flipped',
    ]
    for field_name in copy_fields:
        if field_name in entry:
            new_entry[field_name] = entry[field_name]
    outframes = range(len(entry['clip_ids']))
    # union all the track ids
    all_track_ids = np.array(list(set(gen_utils.flatten_list([
        clip['tracks'].reshape((-1,)).tolist() for clip in entry['clip_ids']]))),
        dtype=entry['tracks'].dtype)
    new_entry['tracks'] = all_track_ids
    ntracks = len(all_track_ids)
    noutframes = len(outframes)
    new_entry['all_frame_ids'] = [clip['frame_id'] for clip in entry['clip_ids']]
    if 'original_file_name' in entry.keys():
        new_entry['original_file_name'] = [
            clip['original_file_name'] for clip in entry['clip_ids']]
    new_entry['gt_keypoints'] = np.zeros((
        ntracks,
        noutframes,
        entry['gt_keypoints'].shape[-2],
        entry['gt_keypoints'].shape[-1]), dtype=entry['gt_keypoints'].dtype)
    new_entry['boxes'] = np.zeros((ntracks, 4 * noutframes), dtype=entry['boxes'].dtype)
    new_entry['is_crowd'] = np.zeros((ntracks,), dtype=entry['is_crowd'].dtype)
    new_entry['gt_overlaps'] = scipy.sparse.csr_matrix(np.zeros(
        (ntracks, entry['gt_overlaps'].shape[1]), dtype=entry['gt_overlaps'].dtype))
    new_entry['gt_classes'] = np.zeros((ntracks,), dtype=entry['gt_classes'].dtype)
    new_entry['track_visible'] = np.full((ntracks, noutframes), False)
    new_entry['segms'] = [[]] * ntracks
    new_entry['box_to_gt_ind_map'] = np.arange(
        ntracks, dtype=entry['box_to_gt_ind_map'].dtype)
    # Just assume 1-class for now => has to be the "person" class
    new_entry['max_classes'] = np.ones(
        (ntracks,), dtype=entry['max_classes'].dtype)
    # All GT, so 1.0 overlap
    new_entry['max_overlaps'] = np.ones((ntracks,), dtype=entry['max_overlaps'].dtype)
    # This isn't really used, so just set to 1s
    new_entry['seg_areas'] = np.ones((ntracks,), dtype=entry['seg_areas'].dtype)
    for cur_track_pos, track_id in enumerate(all_track_ids):
        for cur_frame_pos, frame_id in enumerate(outframes):
            tracks = entry['clip_ids'][frame_id]['tracks'].reshape((-1,)).tolist()
            track_pos = tracks.index(track_id) if track_id in tracks else -1
            if track_pos >= 0:
                new_entry['boxes'][cur_track_pos, cur_frame_pos * 4:
                                   (cur_frame_pos + 1) * 4] = \
                    entry['clip_ids'][frame_id]['boxes'][track_pos]
                new_entry['gt_keypoints'][cur_track_pos, cur_frame_pos, ...] = \
                    entry['clip_ids'][frame_id]['gt_keypoints'][track_pos]
                new_entry['track_visible'][cur_track_pos, cur_frame_pos] = True
                new_entry['gt_classes'][cur_track_pos] = \
                    entry['clip_ids'][frame_id]['gt_classes'][track_pos]
                new_entry['gt_overlaps'][cur_track_pos, 1] = 1.0
    # Since boxes were defined as Nx(4*T+1), I'm modifying the keypoints from
    # NxTx3x17 to Nx3x(17*T). This make the blob dimensions consistent with
    # 2D cases, saving a lot of "if" conditions in the future. Can simply
    # think of predicting 17*T keypoints for a T-length video instead of 17.
    # Also, since 17 is not a fixed number, I can always compute T using
    # the "boxes" entry, as T = boxes.shape[-1] // 4 (a box is always 4 numbers)
    nkpts = new_entry['gt_keypoints'].shape[-1]
    new_entry['gt_keypoints'] = new_entry['gt_keypoints'].transpose(
        (0, 2, 1, 3)).reshape((ntracks, 3, noutframes * nkpts))
    return new_entry


def get_clip(roidb, remove_imperfect=False):
    """
    Add a 'clip_ids' field to each entry of roidb, which contains pointers to
    other elements of roidb that contain the other frames that should go with
    this frame in case of a video.
    """
    video_info = get_video_info(roidb)  # is a dict
    video_info_to_pos = {}
    for el_id, el in video_info.items():
        video_info_to_pos[el] = el_id
    half_T = (cfg.VIDEO.NUM_FRAMES - 1) / 2
    vid_list = range(int(math.floor(-half_T)), int(math.floor(half_T)) + 1)
    assert(len(vid_list) == cfg.VIDEO.NUM_FRAMES)
    assert(vid_list[len(vid_list) // 2] == 0)
    new_roidb = []
    for i, entry in enumerate(tqdm(roidb, desc='Video-fying the roidb')):
        # extract video and frame number information
        this_video_info = video_info[i]
        # collect the clips of T length
        roidb_indexes = [None] * cfg.VIDEO.NUM_FRAMES
        for dt_i, dt in enumerate(vid_list):
            target = (
                this_video_info[0],
                this_video_info[1] + dt * cfg.VIDEO.TIME_INTERVAL,
                this_video_info[2])
            if target in video_info_to_pos.keys():
                pos = video_info_to_pos[target]
                # roidb_indexes.append(
                #     video_info.keys()[video_info.values().index(target)])
                # The pointers are not good enough.. On removing the empty
                # clips the pointers would break
                roidb_indexes[dt_i] = roidb[pos]
        if len([el for el in roidb_indexes if el is None]) > 0 and remove_imperfect:
            continue
        else:
            last_non_none = None
            for k in range(cfg.VIDEO.NUM_FRAMES // 2, -1, -1):
                if roidb_indexes[k] is not None:
                    last_non_none = roidb_indexes[k]
                if roidb_indexes[k] is None:
                    roidb_indexes[k] = last_non_none
            last_non_none = None
            for k in range(cfg.VIDEO.NUM_FRAMES // 2, cfg.VIDEO.NUM_FRAMES):
                if roidb_indexes[k] is not None:
                    last_non_none = roidb_indexes[k]
                if roidb_indexes[k] is None:
                    roidb_indexes[k] = last_non_none
            assert(len([el for el in roidb_indexes if el is None]) == 0)
        entry['clip_ids'] = roidb_indexes
        entry = _combine_clips(entry)
        new_roidb.append(entry)
    logger.info('Video-fied roidb contains {} elements'.format(len(new_roidb)))
    return new_roidb
