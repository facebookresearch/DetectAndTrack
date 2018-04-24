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

import json
import os
import os.path as osp
import glob
from tqdm import tqdm
from PIL import Image
import shutil
import numpy as np

from convert.loader import load_mat
from convert.box import compute_boxes_from_pose
from utils.general import mkdir_p
from convert.data import get_posetrack_kpt_ordering

# Directory with annotation mat files downloaded from PoseTrack website
mat_dir = '/path/to/posetrack_data/annotations/{}'
out_path = '/path/to/output/jsons/posetrack_{}.json'
splits = ['test', 'train', 'val']
# Set this to true if need to re-create the video frames. Note that the
# original frames are in non-standard file format so will need to be fixed.
if 1:  # `convert` the frames to standard format
    recreate_videos = True
    vid_indir = '/path/to/posetrack_data/images'
    vid_outdir = '/path/to/output/images_renamed'
else:  # `convert`-ed frames already exist, do not redo
    recreate_videos = False
    vid_indir = ''
    vid_outdir = '/path/to/output/images_renamed'


def _get_video_info(vpath):
    frame_ids = sorted([int(osp.basename(
        el).split('.')[0]) for el in os.listdir(vpath)])
    nframes = len(frame_ids)
    assert(frame_ids[0] == 1)
    assert(frame_ids[-1] == nframes)
    frame1 = osp.join(vpath, '00000001.jpg')
    wd, ht = Image.open(frame1).size
    return {'nframes': nframes, 'width': wd, 'height': ht}


def _convert_video_frame_ids(inpath, outpath):
    """
    PoseTrack videos follow no consistent naming for frames. Make it consistent
    """
    mkdir_p(outpath)
    frame_names = [osp.basename(el) for el in glob.glob(osp.join(
        inpath, '*.jpg'))]
    # Some videos have 00XX_crop.jpg style filenames
    frame_ids = [int(el.split('.')[0].split('_')[0]) for el in frame_names]
    id_to_name = dict(zip(frame_ids, frame_names))
    for i, fid in enumerate(sorted(frame_ids)):
        shutil.copy('{}/{}'.format(inpath, id_to_name[fid]),
                    '{}/{:08d}.jpg'.format(outpath, i + 1))


def _load_mat_files(annot_dir):
    mat_data = {}
    print('Loading data from MAT files...')
    for fpath in tqdm(glob.glob(osp.join(annot_dir, '*.mat'))):
        stuff = load_mat(fpath)
        if len(stuff) > 0:
            key = osp.dirname(stuff[0].im_name)
            key = key[len('images/'):]
            mat_data[key] = stuff
    return mat_data


def _get_person_category_data():
    category = {
        "supercategory": "person",
        "id": 1,  # to be same as COCO, not using 0
        "name": "person",
        "skeleton": [[16, 14],
                     [14, 12],
                     [17, 15],
                     [15, 13],
                     [12, 13],
                     [6, 12],
                     [7, 13],
                     [6, 7],
                     [6, 8],
                     [7, 9],
                     [8, 10],
                     [9, 11],
                     [2, 3],
                     [1, 2],
                     [1, 3],
                     [2, 4],
                     [3, 5],
                     [4, 6],
                     [5, 7]],
        "keypoints": ["nose",
                      "head_bottom",  # "left_eye",
                      "head_top",  # "right_eye",
                      "left_ear",
                      "right_ear", "left_shoulder", "right_shoulder",
                      "left_elbow", "right_elbow", "left_wrist",
                      "right_wrist", "left_hip", "right_hip", "left_knee",
                      "right_knee", "left_ankle", "right_ankle"]}
    return category


def _get_categories_data():
    return [_get_person_category_data()]


def _gen_image_structure(vname, frame_id, frame_data, vid_info, imid):
    image = {}
    # ordering in data based on tools/video/extract_metadata.py
    image['nframes'] = int(vid_info['nframes'])
    image['frame_id'] = int(frame_id)
    image['width'] = int(vid_info['width'])
    image['height'] = int(vid_info['height'])
    # frames-in-dir kinda videos. The {:08d}.jpg is how I rename it
    image['file_name'] = osp.join(vname, '{:08d}.jpg'.format(frame_id))
    image['original_file_name'] = frame_data.im_name
    image['is_labeled'] = frame_data.is_labeled
    image['id'] = imid
    return image


def _get_posetrack_to_coco_permut():
    print('Computing permutation from posetrack to COCO.')
    target_ordering = _get_person_category_data()['keypoints']
    given_ordering, _ = get_posetrack_kpt_ordering()
    permut_ordering = []
    for given_kpt_id, given_kpt in enumerate(given_ordering):
        new_id = target_ordering.index(given_kpt)
        # Make sure all points get assigned somewhere.
        # COCO has 17 kpts, so the other kpts in posetrack can replace the ones
        # we don't have labels for in posetrack (eye/ear)
        assert(new_id > -1)
        print('{} -> {}'.format(given_kpt_id, new_id))
        permut_ordering.append(new_id)
    return permut_ordering


def _convert_posetrack_kps_to_coco(posetrack_pose, permut_ordering):
    res = np.zeros(
        (len(_get_person_category_data()['keypoints']), 3),
        dtype=posetrack_pose.dtype)
    res[np.array(permut_ordering), :] = posetrack_pose
    return res


def _gen_annot_structure(box_data, kpt_permut_ordering, imid, annid):
    ann = {}
    ann['id'] = annid
    ann['image_id'] = imid
    ann['iscrowd'] = 0
    ann['segmentation'] = []
    ann['num_keypoints'] = 17  # COCO
    ann['category_id'] = 1  # person
    ann['track_id'] = box_data.track_id
    ann['head_box'] = [float(el) for el in box_data.head]
    ann['keypoints'] = _convert_posetrack_kps_to_coco(
        box_data.pose, kpt_permut_ordering).reshape((-1)).tolist()
    ann['bbox'] = compute_boxes_from_pose([[ann['keypoints']]])[0][0]
    ann['area'] = ann['bbox'][-1] * ann['bbox'][-2]
    return ann


def _convert_mat_to_COCO_json(
        annot_dir, out_path, vid_indir, vid_outdir, recreate_videos,
        permut_ordering):
    # Generate the output structure
    res = {}
    res['images'] = []
    res['annotations'] = []
    res['categories'] = _get_categories_data()

    # load all the mat files
    all_annots = _load_mat_files(annot_dir)
    print('Processing MAT files into JSON structures...')
    for vid_name in tqdm(all_annots.keys()):
        # Convert the posetrack video into a sane format
        if recreate_videos:
            assert(len(vid_indir) > 0)
            _convert_video_frame_ids(
                osp.join(vid_indir, vid_name),
                osp.join(vid_outdir, vid_name))
        vid_info = _get_video_info(osp.join(vid_outdir, vid_name))
        vid_data = all_annots[vid_name]
        nframes = len(vid_data)
        for frame_id in range(1, nframes + 1):
            frame_data = vid_data[frame_id - 1]
            image_struct = _gen_image_structure(
                vid_name, frame_id, frame_data, vid_info,
                len(res['images']) + 1)
            res['images'].append(image_struct)
            if frame_data.is_labeled:
                for box_data in frame_data.boxes:
                    annot_struct = _gen_annot_structure(
                        box_data,
                        permut_ordering,
                        imid=len(res['images']),
                        annid=len(res['annotations']) + 1)
                    res['annotations'].append(annot_struct)
    with open(out_path, 'w') as fout:
        json.dump(res, fout)


def main():
    permut_ordering = _get_posetrack_to_coco_permut()
    for split in splits:
        print('Processing {} split'.format(split))
        _convert_mat_to_COCO_json(
            mat_dir.format(split), out_path.format(split),
            vid_indir, vid_outdir, recreate_videos, permut_ordering)


if __name__ == '__main__':
    main()
