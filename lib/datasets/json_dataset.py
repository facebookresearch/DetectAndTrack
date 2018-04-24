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

import os
import numpy as np
import scipy.sparse
import cPickle as pickle
import copy
from tqdm import tqdm
import math

import utils.boxes as box_utils
from utils.timer import Timer
# COCO API
from pycocotools.coco import COCO
from pycocotools import mask as COCOmask

from core.config import cfg
from utils.general import static_vars

import logging
logger = logging.getLogger(__name__)

IM_DIR = 'image_directory'
ANN_FN = 'annotation_file'
# Set to true if the ROIDB needs to be split into frames
SPLIT_INTO_FRAMES = 'split_into_frames'
# Set to true if the frames need to be decoded from videos
FRAMES_FROM_VIDEO = 'frames_from_video'
# Function to read from the weakly labeled outputs
COMPUTED_ANNOTATIONS_INFO = 'computed_annotations_info'
# Optional annotation directory. Used to store additional stuff like for
# jsons for posetrack evaluations
ANN_DN = 'annotation_directory'
DATASETS = {
    'posetrack_v1.0_train': {
        IM_DIR: 'lib/datasets/data/PoseTrack/',
        ANN_FN: 'lib/datasets/lists/PoseTrack/v1.0/posetrack_train.json',
        ANN_DN: 'lib/datasets/data/PoseTrackV1.0_Annots_train_json/',
    },
    'posetrack_v1.0_val': {
        IM_DIR: 'lib/datasets/data/PoseTrack/',
        ANN_FN: 'lib/datasets/lists/PoseTrack/v1.0/posetrack_val.json',
        ANN_DN: 'lib/datasets/data/PoseTrackV1.0_Annots_val_json',
    },
    'posetrack_v1.0_test': {
        IM_DIR: 'lib/datasets/data/PoseTrack/',
        ANN_FN: 'lib/datasets/lists/PoseTrack/v1.0/posetrack_test.json',
        ANN_DN: 'lib/datasets/data/PoseTrackV1.0_Annots_test_json',
    },
}

#### Important conventions for ROIDB
# frame_id: 1-indexed. The reader is 0-indexed, so I make the conversion in
#   utils/image.py


class JsonDataset(object):
    def __init__(self, name):
        assert name in DATASETS.keys(), 'Unknown dataset name'
        logger.debug('Creating: {}'.format(name))
        self.name = name
        self.image_directory = DATASETS[name][IM_DIR]
        self.debug_timer = Timer()
        self.COCO = COCO(DATASETS[name][ANN_FN])
        self.annotation_directory = DATASETS[name][ANN_DN] if ANN_DN in \
                                    DATASETS[name] else ''
        # Set up dataset classes
        category_ids = self.COCO.getCatIds()
        categories = [c['name'] for c in self.COCO.loadCats(category_ids)]
        self.category_to_id_map = dict(zip(categories, category_ids))
        self.classes = ['__background__'] + categories
        self.num_classes = len(self.classes)
        self.json_category_id_to_contiguous_id = {
            v: i + 1 for i, v in enumerate(self.COCO.getCatIds())}
        self.contiguous_category_id_to_json_id = {
            v: k for k, v in self.json_category_id_to_contiguous_id.items()}
        self._init_keypoints(name=self.name)
        # Added by rgirdhar: Used in tracking to know which is head keypoints,
        # when using PCK distance to connect the boxes
        self.person_cat_info = self.COCO.loadCats([
            self.category_to_id_map['person']])[0]
        # Added by rgirdhar: Set true if the frames need to be read out of a
        # video file
        self.frames_from_video = DATASETS[name][FRAMES_FROM_VIDEO] if \
            FRAMES_FROM_VIDEO in DATASETS[name] else False
        self.annotations_info = DATASETS[name][COMPUTED_ANNOTATIONS_INFO] if \
            COMPUTED_ANNOTATIONS_INFO in DATASETS[name] else None
        if self.annotations_info is not None:
            self.annotations_info['clip_length'] = self.annotations_info[
                'clip_length']()

    def get_roidb(
            self, gt=False, proposal_file=None, min_proposal_size=2,
            proposal_limit=-1, crowd_filter_thresh=0):
        assert gt is True or crowd_filter_thresh == 0, \
            'Crowd filter threshold must be 0 if ground-truth annotations ' \
            'are not included.'
        image_ids = self.COCO.getImgIds()
        image_ids.sort()
        roidb = copy.deepcopy(self.COCO.loadImgs(image_ids))
        if len(cfg.ROIDB_SUBSET) > 0:
            roidb = roidb[cfg.ROIDB_SUBSET[0]: cfg.ROIDB_SUBSET[1]]
            logger.warning('Using a roidb subset {}'.format(cfg.ROIDB_SUBSET))
        annots = []
        if SPLIT_INTO_FRAMES in DATASETS[self.name] and DATASETS[
                self.name][SPLIT_INTO_FRAMES]:
            roidb, annots = self._split_roidb_frames(roidb)
        for entry in roidb:
            self._prep_roidb_entry(entry)
        if gt:
            # Include ground-truth object annotations
            self.debug_timer.tic()
            for entry_id, entry in enumerate(roidb):
                self._add_gt_annotations(entry, entry_id, annots)
            logger.debug('_add_gt_annotations took {:.3f}s'.
                         format(self.debug_timer.toc(average=False)))
        if proposal_file is not None:
            # Include proposals from a file
            self.debug_timer.tic()
            self._add_proposals_from_file(
                roidb, proposal_file, min_proposal_size, proposal_limit,
                crowd_filter_thresh)
            logger.debug('_add_proposals_from_file took {:.3f}s'.
                         format(self.debug_timer.toc(average=False)))
        _add_class_assignments(roidb)
        return roidb

    def _prep_roidb_entry(self, entry):
        # Reference back to the parent dataset
        entry['dataset'] = self
        # Make file_name an abs path
        entry['image'] = os.path.join(self.image_directory, entry['file_name'])
        entry['flipped'] = False
        entry['has_visible_keypoints'] = False
        # Empty placeholders
        entry['boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['tracks'] = np.empty((0, 1), dtype=np.int32)
        # head boxes, if available (like in PoseTrack)
        entry['head_boxes'] = np.empty((0, 4), dtype=np.float32)
        entry['segms'] = []
        entry['gt_classes'] = np.empty((0), dtype=np.int32)
        entry['seg_areas'] = np.empty((0), dtype=np.float32)
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(np.empty(
            (0, self.num_classes), dtype=np.float32))
        entry['is_crowd'] = np.empty((0), dtype=np.bool)
        # 'box_to_gt_ind_map': Shape is (#rois). Maps from each roi to the index
        # in the list of rois that satisfy np.where(entry['gt_classes'] > 0)
        entry['box_to_gt_ind_map'] = np.empty((0), dtype=np.int32)
        if self.keypoints is not None:
            entry['gt_keypoints'] = np.empty(
                (0, 3, self.num_keypoints), dtype=np.int32)
        # Remove unwanted fields if they exist
        for k in ['date_captured', 'url', 'license', 'file_name']:
            if k in entry:
                del entry[k]

    def convert_raw_predictions_to_objs(self, annots, image_id):
        if len(annots['boxes']) == 0:
            return []
        objs = []
        N = annots['boxes'].shape[0]
        for i in range(N):
            obj = {}
            # COCO labels are in xywh format, but I make predictions in xyxy
            # Remove the score from box before converting
            obj['bbox'] = box_utils.xyxy_to_xywh(annots['boxes'][i][
                np.newaxis, :4]).reshape((-1,)).tolist()
            obj['num_keypoints'] = annots['poses'][i].shape[-1]
            assert(obj['num_keypoints'] == cfg.KRCNN.NUM_KEYPOINTS)
            obj['segmentation'] = []
            obj['area'] = obj['bbox'][-1] * obj['bbox'][-2]
            obj['iscrowd'] = False
            pose = annots['poses'][i][:3].transpose()
            pose[pose[:, -1] >= 2.0, -1] = 2
            pose[pose[:, -1] < 2.0, -1] = 0
            obj['keypoints'] = pose.reshape((-1)).tolist()
            obj['track_id'] = annots['tracks'][i]
            obj['image_id'] = image_id
            obj['category_id'] = 1  # person
            objs.append(obj)
        return objs

    def _add_gt_annotations(self, entry, entry_id, annots):
        if len(annots) > 0:
            objs = self.convert_raw_predictions_to_objs(
                annots[entry_id], entry['id'])
        else:
            ann_ids = self.COCO.getAnnIds(imgIds=entry['id'], iscrowd=None)
            objs = self.COCO.loadAnns(ann_ids)
        # Sanitize bboxes -- some are invalid
        valid_objs = []
        valid_segms = []
        width = entry['width']
        height = entry['height']
        for obj in objs:
            # crowd regions are RLE encoded and stored as dicts
            if isinstance(obj['segmentation'], list):
                # Valid polygons have >= 3 points, so require >= 6 coordinates
                obj['segmentation'] = [
                    p for p in obj['segmentation'] if len(p) >= 6
                ]
            if obj['area'] < cfg.TRAIN.GT_MIN_AREA:
                continue
            if 'ignore' in obj and obj['ignore'] == 1:
                continue
            # Convert form x1, y1, w, h to x1, y1, x2, y2
            x1 = obj['bbox'][0]
            y1 = obj['bbox'][1]
            x2 = x1 + np.maximum(0., obj['bbox'][2] - 1.)
            y2 = y1 + np.maximum(0., obj['bbox'][3] - 1.)
            x1, y1, x2, y2 = box_utils.clip_xyxy_to_image(
                x1, y1, x2, y2, height, width)
            # Require non-zero seg area and more than 1x1 box size
            if obj['area'] > 0 and x2 > x1 and y2 > y1:
                obj['clean_bbox'] = [x1, y1, x2, y2]
                valid_objs.append(obj)
                valid_segms.append(obj['segmentation'])
        num_valid_objs = len(valid_objs)

        boxes = np.zeros((num_valid_objs, 4), dtype=entry['boxes'].dtype)
        tracks = -np.ones((num_valid_objs, 1), dtype=entry['tracks'].dtype)
        head_boxes = -np.ones((num_valid_objs, 4),
                              dtype=entry['head_boxes'].dtype)
        gt_classes = np.zeros((num_valid_objs), dtype=entry['gt_classes'].dtype)
        gt_overlaps = np.zeros(
            (num_valid_objs, self.num_classes),
            dtype=entry['gt_overlaps'].dtype)
        seg_areas = np.zeros((num_valid_objs), dtype=entry['seg_areas'].dtype)
        is_crowd = np.zeros((num_valid_objs), dtype=entry['is_crowd'].dtype)
        box_to_gt_ind_map = np.zeros(
            (num_valid_objs), dtype=entry['box_to_gt_ind_map'].dtype)
        if self.keypoints is not None:
            gt_keypoints = np.zeros(
                (num_valid_objs, 3, self.num_keypoints),
                dtype=entry['gt_keypoints'].dtype)

        im_has_visible_keypoints = False
        for ix, obj in enumerate(valid_objs):
            cls = self.json_category_id_to_contiguous_id[obj['category_id']]
            boxes[ix, :] = obj['clean_bbox']
            if 'track_id' in obj:
                tracks[ix, 0] = obj['track_id']
            if 'head_box' in obj:
                # NOTE: This box has NOT BEEN CLEANED, and NOT BEEN converted
                # to (xmin, ymin, xmax, ymax). This is only here to be used
                # in MPII evaluations
                head_boxes[ix, :] = obj['head_box']
            gt_classes[ix] = cls
            seg_areas[ix] = obj['area']
            is_crowd[ix] = obj['iscrowd']
            box_to_gt_ind_map[ix] = ix
            if self.keypoints is not None:
                gt_keypoints[ix, :, :] = self._get_gt_keypoints(obj)
                if np.sum(gt_keypoints[ix, 2, :]) > 0:
                    im_has_visible_keypoints = True
            if obj['iscrowd']:
                # Set overlap to -1 for all classes for crowd objects
                # so they will be excluded during training
                gt_overlaps[ix, :] = -1.0
            else:
                gt_overlaps[ix, cls] = 1.0
        entry['boxes'] = np.append(entry['boxes'], boxes, axis=0)
        entry['tracks'] = np.append(entry['tracks'], tracks, axis=0)
        entry['head_boxes'] = np.append(entry['head_boxes'], head_boxes, axis=0)
        entry['segms'].extend(valid_segms)
        # To match the original implementation:
        # entry['boxes'] = np.append(
        #     entry['boxes'], boxes.astype(np.int).astype(np.float), axis=0)
        entry['gt_classes'] = np.append(entry['gt_classes'], gt_classes)
        entry['seg_areas'] = np.append(entry['seg_areas'], seg_areas)
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0)
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(entry['is_crowd'], is_crowd)
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'], box_to_gt_ind_map)
        if self.keypoints is not None:
            entry['gt_keypoints'] = np.append(
                entry['gt_keypoints'], gt_keypoints, axis=0)
            entry['has_visible_keypoints'] = im_has_visible_keypoints

    def _add_proposals_from_file(
            self, roidb, proposal_file, min_proposal_size, top_k, crowd_thresh):
        logger.info('Loading proposals from: {}'.format(proposal_file))
        with open(proposal_file, 'r') as f:
            proposals = pickle.load(f)
        id_field = 'indexes' if 'indexes' in proposals else 'ids'  # compat fix
        _sort_proposals(proposals, id_field)
        box_list = []
        for i, entry in enumerate(roidb):
            if i % 2500 == 0:
                logger.info(' {:d}/{:d}'.format(i + 1, len(roidb)))
            boxes = proposals['boxes'][i]
            # Sanity check that these boxes are for the correct image id
            assert entry['id'] == proposals[id_field][i]
            # Remove duplicate boxes and very small boxes and then take top k
            boxes = box_utils.clip_boxes_to_image(
                boxes, entry['height'], entry['width'])
            keep = box_utils.unique_boxes(boxes)
            boxes = boxes[keep, :]
            keep = box_utils.filter_small_boxes(boxes, min_proposal_size)
            boxes = boxes[keep, :]
            if top_k > 0:
                boxes = boxes[:top_k, :]
            box_list.append(boxes)
        _merge_proposal_boxes_into_roidb(roidb, box_list)
        if crowd_thresh > 0:
            _filter_crowd_proposals(roidb, crowd_thresh)

    def _init_keypoints(self, name=''):
        self.keypoints = None
        self.keypoint_flip_map = None
        self.keypoints_to_id_map = None
        self.num_keypoints = 0
        # Thus far only the 'person' category has keypoints
        if 'person' in self.category_to_id_map:
            cat_info = self.COCO.loadCats([self.category_to_id_map['person']])
        else:
            return

        # Check if the annotations contain keypoint data or not
        if 'keypoints' in cat_info[0]:
            keypoints = cat_info[0]['keypoints']
            self.keypoints_to_id_map = dict(
                zip(keypoints, range(len(keypoints))))
            self.keypoints = keypoints
            self.num_keypoints = len(keypoints)
            if name.startswith('keypoints_coco'):
                self.keypoint_flip_map = {
                    'left_eye': 'right_eye',
                    'left_ear': 'right_ear',
                    'left_shoulder': 'right_shoulder',
                    'left_elbow': 'right_elbow',
                    'left_wrist': 'right_wrist',
                    'left_hip': 'right_hip',
                    'left_knee': 'right_knee',
                    'left_ankle': 'right_ankle'}
            else:
                self.keypoint_flip_map = {
                    'left_shoulder': 'right_shoulder',
                    'left_elbow': 'right_elbow',
                    'left_wrist': 'right_wrist',
                    'left_hip': 'right_hip',
                    'left_knee': 'right_knee',
                    'left_ankle': 'right_ankle'}

    def _get_gt_keypoints(self, obj):
        if 'keypoints' not in obj:
            return None
        kp = np.array(obj['keypoints'])
        x = kp[0::3]  # 0-indexed x coordinates
        y = kp[1::3]  # 0-indexed y coordinates
        # 0: not labeled; 1: labeled, not inside mask;
        # 2: labeled and inside mask
        v = kp[2::3]
        num_keypoints = len(obj['keypoints']) / 3
        assert num_keypoints == self.num_keypoints
        gt_kps = np.ones((3, self.num_keypoints), dtype=np.int32)
        for i in range(self.num_keypoints):
            gt_kps[0, i] = x[i]
            gt_kps[1, i] = y[i]
            gt_kps[2, i] = v[i]
        return gt_kps

    def _split_roidb_frames(self, roidb):
        # Config options
        clips_per_video = cfg.VIDEO.DEFAULT_CLIPS_PER_VIDEO
        clip_length = 1  # 1-frame clips
        if self.annotations_info is not None:
            clips_per_video = self.annotations_info['clips_per_video']
            clip_length = self.annotations_info['clip_length']
            entry_to_shard = _assign_shard_id_to_roidb(
                roidb, self.annotations_info['num_splits'],
                self.annotations_info['tot_vids'])

        # For each video in roidb, split into a entry per-frame
        new_roidb = []
        new_annots = []
        for entry_id, entry in enumerate(tqdm(roidb, desc='Splitting video->frames')):
            assert 'nframes' in entry, 'Video dataset must have nframes'
            # Get annotations, if possible
            annots = {}
            if self.annotations_info is not None:
                annots = _read_weak_annotations(
                    entry_to_shard[entry_id],
                    data_dir=self.annotations_info['data_dir'],
                    det_file_name=self.annotations_info['det_file_name'])
                assert(len(annots['boxes']) == entry['nframes'])
            # roidb frame_ids are 1-indexed
            already_added = {}  # don't add same frame multiple times
            step_size = max(entry['nframes'] // clips_per_video, 1)
            for start_frame_id in range(1, entry['nframes'] + 2 - clip_length,
                                        step_size):
                for frame_id in range(start_frame_id, start_frame_id + clip_length):
                    if frame_id in already_added:
                        continue
                    new_entry = copy.deepcopy(entry)
                    new_entry['frame_id'] = frame_id
                    new_roidb.append(new_entry)
                    if len(annots) != 0:
                        new_annots.append({
                            # frame_id is 1-indexed
                            'boxes': annots['boxes'][frame_id - 1],
                            'poses': annots['poses'][frame_id - 1],
                            'tracks': annots['tracks'][frame_id - 1],
                        })
                    already_added[frame_id] = True
        logger.info('New roidb size {}'.format(len(new_roidb)))
        return new_roidb, new_annots


def _merge_proposal_boxes_into_roidb(roidb, box_list):
    assert len(box_list) == len(roidb)
    for i, entry in enumerate(roidb):
        boxes = box_list[i]
        num_boxes = boxes.shape[0]
        gt_overlaps = np.zeros(
            (num_boxes, entry['gt_overlaps'].shape[1]),
            dtype=entry['gt_overlaps'].dtype)
        box_to_gt_ind_map = -np.ones(
            (num_boxes), dtype=entry['box_to_gt_ind_map'].dtype)

        # Note: unlike in other places, here we intentionally include all gt
        # rois, even ones marked as crowd. Boxes that overlap with crowds will
        # be filtered out later (see: _filter_crowd_proposals).
        gt_inds = np.where(entry['gt_classes'] > 0)[0]
        if len(gt_inds) > 0:
            gt_boxes = entry['boxes'][gt_inds, :]
            gt_classes = entry['gt_classes'][gt_inds]
            proposal_to_gt_overlaps = box_utils.bbox_overlaps(
                boxes.astype(dtype=np.float32, copy=False),
                gt_boxes.astype(dtype=np.float32, copy=False))
            # Gt box that overlaps each input box the most
            # (ties are broken arbitrarily by class order)
            argmaxes = proposal_to_gt_overlaps.argmax(axis=1)
            # Amount of that overlap
            maxes = proposal_to_gt_overlaps.max(axis=1)
            # Those boxes with non-zero overlap with gt boxes
            I = np.where(maxes > 0)[0]
            # Record max overlaps with the class of the appropriate gt box
            gt_overlaps[I, gt_classes[argmaxes[I]]] = maxes[I]
            box_to_gt_ind_map[I] = gt_inds[argmaxes[I]]
        entry['boxes'] = np.append(
            entry['boxes'],
            boxes.astype(entry['boxes'].dtype, copy=False),
            axis=0)
        entry['gt_classes'] = np.append(
            entry['gt_classes'],
            np.zeros((num_boxes), dtype=entry['gt_classes'].dtype))
        entry['seg_areas'] = np.append(
            entry['seg_areas'],
            np.zeros((num_boxes), dtype=entry['seg_areas'].dtype))
        entry['gt_overlaps'] = np.append(
            entry['gt_overlaps'].toarray(), gt_overlaps, axis=0)
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(entry['gt_overlaps'])
        entry['is_crowd'] = np.append(
            entry['is_crowd'],
            np.zeros((num_boxes), dtype=entry['is_crowd'].dtype))
        entry['box_to_gt_ind_map'] = np.append(
            entry['box_to_gt_ind_map'],
            box_to_gt_ind_map.astype(
                entry['box_to_gt_ind_map'].dtype, copy=False))


def _filter_crowd_proposals(roidb, crowd_thresh):
    """Finds proposals that are inside crowd regions and marks them as
    overlap = -1 with each ground-truth rois, which means they will be excluded
    from training.
    """
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        crowd_inds = np.where(entry['is_crowd'] == 1)[0]
        non_gt_inds = np.where(entry['gt_classes'] == 0)[0]
        if len(crowd_inds) == 0 or len(non_gt_inds) == 0:
            continue
        crowd_boxes = box_utils.xyxy_to_xywh(entry['boxes'][crowd_inds, :])
        non_gt_boxes = box_utils.xyxy_to_xywh(entry['boxes'][non_gt_inds, :])
        iscrowd_flags = [int(True)] * len(crowd_inds)
        ious = COCOmask.iou(non_gt_boxes, crowd_boxes, iscrowd_flags)
        bad_inds = np.where(ious.max(axis=1) > crowd_thresh)[0]
        gt_overlaps[non_gt_inds[bad_inds], :] = -1
        entry['gt_overlaps'] = scipy.sparse.csr_matrix(gt_overlaps)


def _add_class_assignments(roidb):
    for entry in roidb:
        gt_overlaps = entry['gt_overlaps'].toarray()
        # max overlap with gt over classes (columns)
        max_overlaps = gt_overlaps.max(axis=1)
        # gt class that had the max overlap
        max_classes = gt_overlaps.argmax(axis=1)
        entry['max_classes'] = max_classes
        entry['max_overlaps'] = max_overlaps
        # sanity checks
        # if max overlap is 0, the class must be background (class 0)
        zero_inds = np.where(max_overlaps == 0)[0]
        assert all(max_classes[zero_inds] == 0)
        # if max overlap > 0, the class must be a fg class (not class 0)
        nonzero_inds = np.where(max_overlaps > 0)[0]
        assert all(max_classes[nonzero_inds] != 0)


def _sort_proposals(proposals, id_field):
    order = np.argsort(proposals[id_field])
    fields_to_sort = ['boxes', id_field, 'scores']
    for k in fields_to_sort:
        proposals[k] = [proposals[k][i] for i in order]


def add_proposals(roidb, rois, scales):
    """Add proposal boxes (rois) to an roidb that has ground-truth annotations
    but no proposals. If the proposals are not at the original image scale,
    specify the scale factor that separate them in scales.
    """
    box_list = []
    for i in range(len(roidb)):
        inv_im_scale = 1. / scales[i]
        idx = np.where(rois[:, 0] == i)[0]
        box_list.append(rois[idx, 1:] * inv_im_scale)
    _merge_proposal_boxes_into_roidb(roidb, box_list)
    # For historical consistency, not filter crowds (TODO(rbg): investigate)
    # json_dataset._filter_crowd_proposals(roidb, cfg.TRAIN.CROWD_FILTER_THRESH)
    _add_class_assignments(roidb)


def _assign_shard_id_to_roidb(roidb, num_splits, tot_vids):
    """
    Returns:
        list with one element for each entry in roidb
        (shard_dir_name,
            (start_frame_id (0-indexed, included),
             end_frame_id (0-indexed, not included)))
    """
    shards = []
    vids_per_job = int(math.ceil(tot_vids / num_splits))
    last_proc = 0
    for start_id in range(num_splits):
        this_end_pos = min(last_proc + vids_per_job, tot_vids + 1)
        this_outdir = '{0:05d}_range_{1}_{2}'.format(
            start_id, last_proc, this_end_pos)
        # run through the entries that get assigned to this shard, and set
        # what frames out of it belong to which video.
        last_frame_proc = 0
        for i in range(last_proc, min(this_end_pos, len(roidb))):
            # start_id is included and last_proc is not, as happens in the
            # ROIDB_SUBSET code
            this_frame_proc = last_frame_proc + roidb[i]['nframes']
            shards.append((
                this_outdir, (last_frame_proc, this_frame_proc)))
            last_frame_proc = this_frame_proc
        last_proc = this_end_pos
    return shards


def pickle_cached_load(fpath, cache):
    if fpath in cache:
        return cache[fpath]
    with open(fpath, 'r') as fin:
        data = pickle.load(fin)
    cache.clear()
    cache[fpath] = data
    return data


@static_vars(weak_annot_cache={})
def _read_weak_annotations(shard_info, data_dir='',
                           det_file_name='detections.pkl',
                           fixed_str='test/kinetics_unlabeled_train/keypoint_rcnn'):
    det_fpath = os.path.join(data_dir, shard_info[0], fixed_str, det_file_name)
    data = pickle_cached_load(det_fpath, _read_weak_annotations.weak_annot_cache)
    boxes = data['all_boxes'][1][shard_info[1][0]: shard_info[1][1]]
    poses = data['all_keyps'][1][shard_info[1][0]: shard_info[1][1]]
    tracks = data['all_tracks'][1][shard_info[1][0]: shard_info[1][1]]
    assert(len(boxes) == len(poses))
    assert(len(boxes) == len(tracks))
    return {'boxes': boxes, 'poses': poses, 'tracks': tracks}
