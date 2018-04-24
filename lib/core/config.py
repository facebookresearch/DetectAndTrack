##############################################################
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################

"""Fast R-CNN config system.

This file specifies default config options for Fast R-CNN. You should not
change values in this file. Instead, you should write a config file (in yaml)
and use cfg_from_file(yaml_file) to load it and override the default options.

Most tools in $ROOT/tools take a --cfg option to specify an override file.
    - See tools/{train,test}_net.py for example code that uses cfg_from_file()
    - See experiments/cfgs/*.yml for example YAML config override files
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import copy
import os
import os.path as osp
import numpy as np
from utils.collections import AttrDict

import logging
logger = logging.getLogger(__name__)

__C = AttrDict()
# Consumers can get config by:
#   from core.config import cfg
cfg = __C


# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
__C.TRAIN = AttrDict()

# Initialize network with weights from this pickle file
__C.TRAIN.WEIGHTS = b''

# Dataset to use
__C.TRAIN.DATASET = b''

# Scales to use during training (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TRAIN.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TRAIN.MAX_SIZE = 1000

# Images to use per minibatch
__C.TRAIN.IMS_PER_BATCH = 2

# Minibatch size (number of regions of interest [ROIs])
__C.TRAIN.BATCH_SIZE_PER_IM = 64

# Fraction of minibatch that is labeled foreground (i.e. class > 0)
__C.TRAIN.FG_FRACTION = 0.25

# Overlap threshold for a ROI to be considered foreground (if >= FG_THRESH)
__C.TRAIN.FG_THRESH = 0.5

# Overlap threshold for a ROI to be considered background (class = 0 if
# overlap in [LO, HI))
__C.TRAIN.BG_THRESH_HI = 0.5
__C.TRAIN.BG_THRESH_LO = 0.0

# Use horizontally-flipped images during training?
__C.TRAIN.USE_FLIPPED = True

# Train bounding-box regressors
# This only refers to the Fast R-CNN bbox reg used to transform proposals
# This does not refer to the bbox reg that happens in RPN
__C.TRAIN.BBOX_REG = True

# Overlap required between a ROI and ground-truth box in order for that ROI to
# be used as a bounding-box regression training example
__C.TRAIN.BBOX_THRESH = 0.5

# Unnormalized iterations between snapshots
# Divided by NUM_GPUS to determine frequency
__C.TRAIN.SNAPSHOT_ITERS = 20000

# Train using these proposals
__C.TRAIN.PROPOSAL_FILE = b''

# Make minibatches from images that have similar aspect ratios (i.e. both
# tall and thin or both short and wide) in order to avoid wasting computation
# on zero-padding.
__C.TRAIN.ASPECT_GROUPING = True

# IOU >= thresh: positive example
__C.TRAIN.RPN_POSITIVE_OVERLAP = 0.7
# IOU < thresh: negative example
__C.TRAIN.RPN_NEGATIVE_OVERLAP = 0.3
# Max number of foreground examples
__C.TRAIN.RPN_FG_FRACTION = 0.5
# Total number of examples
__C.TRAIN.RPN_BATCH_SIZE_PER_IM = 256
# NMS threshold used on RPN proposals
__C.TRAIN.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TRAIN.RPN_PRE_NMS_TOP_N = 12000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TRAIN.RPN_POST_NMS_TOP_N = 2000
# By default RPN anchors are removed while training if they go outside the image
# by RPN_STRADDLE_THRESH pixels (set to -1 or a large value, e.g. 100000, to
# disable pruning)
__C.TRAIN.RPN_STRADDLE_THRESH = 0
# Proposal height and width both need to be greater than RPN_MIN_SIZE
# (at orig image scale) -- only used by end-to-end Faster R-CNN training
__C.TRAIN.RPN_MIN_SIZE = 0
# Dropout rate for FC layers (<= 0 means no dropout)
__C.TRAIN.DROPOUT = 0.0
__C.TRAIN.CROWD_FILTER_THRESH = 0.7
# Ignore ground-truth objects with area < this threshold
__C.TRAIN.GT_MIN_AREA = -1
# Prefetch Queue size
__C.TRAIN.MINIBATCH_QUEUE_SIZE = 64


# ---------------------------------------------------------------------------- #
# Inference options
# ---------------------------------------------------------------------------- #
__C.TEST = AttrDict()

# Initialize network with weights from this pickle file
__C.TEST.WEIGHTS = b''

# Dataset to use
__C.TEST.DATASET = b''
# Alternatively, a tuple of datasets can be specified
__C.TEST.DATASETS = ()

# Scales to use during testing (can list multiple scales)
# Each scale is the pixel size of an image's shortest side
__C.TEST.SCALES = (600,)

# Max pixel size of the longest side of a scaled input image
__C.TEST.MAX_SIZE = 1000

# Overlap threshold used for non-maximum suppression (suppress boxes with
# IoU >= this threshold)
__C.TEST.NMS = 0.3

__C.TEST.SOFT_NMS = AttrDict()
__C.TEST.SOFT_NMS.ENABLED = False
__C.TEST.SOFT_NMS.METHOD = b'linear'
__C.TEST.SOFT_NMS.SIGMA = 0.5
# For the overlap threshold, we use TEST.NMS

# Bounding Box Voting from the Multi-Region CNN paper
__C.TEST.BBOX_VOTE = AttrDict()
__C.TEST.BBOX_VOTE.ENABLED = False
# We use TEST.NMS threshold for the NMS step. VOTE_TH overlap threshold
# is used to select voting boxes (IoU >= VOTE_TH) for each box that survives NMS
__C.TEST.BBOX_VOTE.VOTE_TH = 0.8

# Experimental: treat the (K+1) units in the cls_score layer as linear
# predictors (trained, eg, with one-vs-rest SVMs).
__C.TEST.SVM = False

# Test using bounding-box regressors
__C.TEST.BBOX_REG = True

# Test using these proposals
__C.TEST.PROPOSAL_FILE = b''
# Alternatively, a tuple of proposal files can be specified (must be used with
# TEST.DATASETS)
__C.TEST.PROPOSAL_FILES = ()
__C.TEST.PROPOSAL_LIMIT = 2000

# NMS threshold used on RPN proposals
__C.TEST.RPN_NMS_THRESH = 0.7
# Number of top scoring boxes to keep before apply NMS to RPN proposals
__C.TEST.RPN_PRE_NMS_TOP_N = 12000
# Number of top scoring boxes to keep after applying NMS to RPN proposals
__C.TEST.RPN_POST_NMS_TOP_N = 2000
# Proposal height and width both need to be greater than RPN_MIN_SIZE
# (at orig image scale)
__C.TEST.RPN_MIN_SIZE = 0
__C.TEST.DETECTIONS_PER_IM = 100
__C.TEST.SCORE_THRESH = 0.05
__C.TEST.COMPETITION_MODE = True
# Can set to True to force PASCAL to be evaluated with the COCO eval code
__C.TEST.FORCE_JSON_DATASET_EVAL = False

# Test-time augmentations for bounding box detection
__C.TEST.BBOX_AUG = AttrDict()
# Heuristic used to combine predicted box scores
#   Valid options: ('ID', 'AVG', 'UNION')
__C.TEST.BBOX_AUG.SCORE_HEUR = b'ID'
# Heuristic used to combine predicted box coordinates
#   Valid options: ('ID', 'AVG', 'UNION')
__C.TEST.BBOX_AUG.COORD_HEUR = b'ID'
# Horizontal flip at the original scale (id transform)
__C.TEST.BBOX_AUG.H_FLIP = False
# Each scale is the pixel size of an image's shortest side
__C.TEST.BBOX_AUG.SCALES = ()
# Max pixel size of the longer side
__C.TEST.BBOX_AUG.MAX_SIZE = 4000
# Horizontal flip at each scale
__C.TEST.BBOX_AUG.SCALE_H_FLIP = False
# Apply scaling based on object size
__C.TEST.BBOX_AUG.SCALE_SIZE_DEP = False
__C.TEST.BBOX_AUG.AREA_TH_LO = 50**2
__C.TEST.BBOX_AUG.AREA_TH_HI = 180**2
# Each aspect ratio is relative to image width
__C.TEST.BBOX_AUG.ASPECT_RATIOS = ()
# Horizontal flip at each aspect ratio
__C.TEST.BBOX_AUG.ASPECT_RATIO_H_FLIP = False

# Test-time augmentations for mask detection
__C.TEST.MASK_AUG = AttrDict()
# Heuristic used to combine mask predictions
# SOFT prefix indicates that the computation is performed on soft masks
#   Valid options: ('SOFT_AVG', 'SOFT_MAX', 'LOGIT_AVG')
__C.TEST.MASK_AUG.HEUR = b'SOFT_AVG'
# Horizontal flip at the original scale (id transform)
__C.TEST.MASK_AUG.H_FLIP = False
# Each scale is the pixel size of an image's shortest side
__C.TEST.MASK_AUG.SCALES = ()
# Max pixel size of the longer side
__C.TEST.MASK_AUG.MAX_SIZE = 4000
# Horizontal flip at each scale
__C.TEST.MASK_AUG.SCALE_H_FLIP = False
# Apply scaling based on object size
__C.TEST.MASK_AUG.SCALE_SIZE_DEP = False
__C.TEST.MASK_AUG.AREA_TH = 180**2
# Each aspect ratio is relative to image width
__C.TEST.MASK_AUG.ASPECT_RATIOS = ()
# Horizontal flip at each aspect ratio
__C.TEST.MASK_AUG.ASPECT_RATIO_H_FLIP = False

# Test-augmentations for keypoints detection
__C.TEST.KPS_AUG = AttrDict()
# Heuristic used to combine keypoint predictions
#   Valid options: ('HM_AVG', 'HM_MAX')
__C.TEST.KPS_AUG.HEUR = b'HM_AVG'
# Horizontal flip at the original scale (id transform)
__C.TEST.KPS_AUG.H_FLIP = False
# Each scale is the pixel size of an image's shortest side
__C.TEST.KPS_AUG.SCALES = ()
# Max pixel size of the longer side
__C.TEST.KPS_AUG.MAX_SIZE = 4000
# Horizontal flip at each scale
__C.TEST.KPS_AUG.SCALE_H_FLIP = False
# Apply scaling based on object size
__C.TEST.KPS_AUG.SCALE_SIZE_DEP = False
__C.TEST.KPS_AUG.AREA_TH = 180**2
# Eeach aspect ratio is realtive to image width
__C.TEST.KPS_AUG.ASPECT_RATIOS = ()
# Horizontal flip at each aspect ratio
__C.TEST.KPS_AUG.ASPECT_RATIO_H_FLIP = False

# Model ensembling
__C.TEST.ENSEMBLE = AttrDict()
# Cache results on the devstorage node. When this option is enabled,
# cache paths can be specified relative to the devstorage remote path
__C.TEST.ENSEMBLE.DEVSTORAGE_CACHE = False
# RPN configs to use for computing proposals
__C.TEST.ENSEMBLE.RPN_CONFIGS = ()
# Directory where proposals files are cached
__C.TEST.ENSEMBLE.PROPOSAL_CACHE = b'/tmp'
# Set to true to initialize the model with random weights before loading weights
# from the pkl file. This is useful when testing a 3D model with a 2D model,
# and can use the inflation machinery. However, can also lead to bugs when
# certain blobs are not present in the trained model (pkl) but the testing
# will not complain as it will keep using the randomly initialized blobs,
# so be careful.
__C.TEST.INIT_RANDOM_VARS_BEFORE_LOADING = False
# Extract CNN features. Will not run Mask R-CNN, but will simply extract feats
# using pytorch models
__C.TEST.EXT_CNN_FEATURES = False
# Can be 'ImNet' or 'Places'
__C.TEST.EXT_CNN_FEATURES_MODEL = b'ImNet'


# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
__C.MODEL = AttrDict()
__C.MODEL.TYPE = b''
__C.MODEL.CONV_BODY = b''
__C.MODEL.ROI_HEAD = b''   # TODO: rename to BBOX_ROI_HEAD?
__C.MODEL.NUM_CLASSES = -1
__C.MODEL.PS_GRID_SIZE = 3  # TODO: move to .RFCN.PS_GRID_SIZE
__C.MODEL.DILATION = 1  # TODO: move to .RFCN.DILATION
__C.MODEL.CLS_AGNOSTIC_BBOX_REG = False
__C.MODEL.RPN_ONLY = False
__C.MODEL.FASTER_RCNN = False
__C.MODEL.MASK_ON = False
__C.MODEL.KEYPOINTS_ON = False
# Use 'prof_dag' to get profiling statistics
__C.MODEL.EXECUTION_TYPE = b'dag'
# Default weights on (dx, dy, dw, dh) for normalizing bbox regression targets
# These are empirically chosen to approximately lead to unit variance targets
__C.MODEL.BBOX_REG_WEIGHTS = (10., 10., 5., 5.)

__C.MODEL.VIDEO_ON = False

## Batch normalization

# If true, use the SpatialBN layer instead of AffineChannel layer
__C.MODEL.USE_BN = False
# If true, use the BN layer in test mode (i.e. should be same output as the
# Affine layer).
__C.MODEL.USE_BN_TESTMODE_ONLY = False
# From Kaiming's Halekala
__C.MODEL.BN_EPSILON = 1.0000001e-5
__C.MODEL.BN_MOMENTUM = 0.9

# ---------------------------------------------------------------------------- #
# Solver options
# ---------------------------------------------------------------------------- #
__C.SOLVER = AttrDict()
__C.SOLVER.BASE_LR = 0.001
__C.SOLVER.LR_POLICY = b'step'
__C.SOLVER.GAMMA = 0.1
__C.SOLVER.STEP_SIZE = 30000
__C.SOLVER.MAX_ITER = 40000
__C.SOLVER.MOMENTUM = 0.9
__C.SOLVER.WEIGHT_DECAY = 0.0005
__C.SOLVER.WARM_UP_ITERS = 500
__C.SOLVER.WARM_UP_FACTOR = 1.0 / 3.0
# WARM_UP_METHOD can be either 'constant' or 'linear'
__C.SOLVER.WARM_UP_METHOD = 'linear'
__C.SOLVER.STEPS = []
__C.SOLVER.LRS = []
# Scale the momentum update history by new_lr / old_lr when updating the
# learning rate (this is correct given MomentumSGDUpdateOp)
__C.SOLVER.SCALE_MOMENTUM = True
__C.SOLVER.SCALE_MOMENTUM_THRESHOLD = 1.1
__C.SOLVER.LOG_LR_CHANGE_THRESHOLD = 1.1
# LR Policies (by example):
# 'step'
#   lr = BASE_LR * GAMMA ** (cur_iter // STEP_SIZE)
# 'steps_with_decay'
#   SOLVER.STEPS = [0, 60000, 80000]
#   SOLVER.GAMMA = 0.1
#   lr = BASE_LR * GAMMA ** current_step
#   iters [0, 59999] are in current_step = 0, iters [60000, 79999] are in
#   current_step = 1, and so on
# 'steps_with_lrs'
#   SOLVER.STEPS = [0, 60000, 80000]
#   SOLVER.LRS = [0.02, 0.002, 0.0002]
#   lr = LRS[current_step]


# ---------------------------------------------------------------------------- #
# Fast R-CNN options
# ---------------------------------------------------------------------------- #
__C.FAST_RCNN = AttrDict()
__C.FAST_RCNN.MLP_HEAD_DIM = 1024
__C.FAST_RCNN.ROI_XFORM_METHOD = b'RoIPoolF'
# Only applies to RoIWarp, RoIWarpMax, and RoIAlign
__C.FAST_RCNN.ROI_XFORM_SAMPLING_RATIO = 0
# Models may ignore this and use fixed values
__C.FAST_RCNN.ROI_XFORM_RESOLUTION = 14


# ---------------------------------------------------------------------------- #
# RPN options
# ---------------------------------------------------------------------------- #
__C.RPN = AttrDict()
__C.RPN.ON = False
# Note: these options are *not* used by FPN RPN; see FPN.RPN* options
# RPN anchor sizes
__C.RPN.SIZES = (64, 128, 256, 512)
# Stride of the feature map that RPN is attached to
__C.RPN.STRIDE = 16
# RPN anchor aspect ratios
__C.RPN.ASPECT_RATIOS = (0.5, 1, 2)


# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
__C.FPN = AttrDict()
__C.FPN.FPN_ON = False  # avoid using 'ON', yaml converts it to True
__C.FPN.DIM = 256
__C.FPN.ZERO_INIT_LATERAL = False
__C.FPN.COARSEST_STRIDE = 32
# Multilevel RoI transform
__C.FPN.MULTILEVEL_ROIS = False
__C.FPN.ROI_CANONICAL_SCALE = 224  # s0
__C.FPN.ROI_CANONICAL_LEVEL = 4    # k0: where s0 maps to
__C.FPN.ROI_MAX_LEVEL = 5  # coarsest level of pyramid
__C.FPN.ROI_MIN_LEVEL = 2  # finest level of pyramid
# Multilevel RPN
__C.FPN.MULTILEVEL_RPN = False
__C.FPN.RPN_MAX_LEVEL = 6
__C.FPN.RPN_MIN_LEVEL = 2
# FPN RPN anchor aspect ratios
__C.FPN.RPN_ASPECT_RATIOS = (0.5, 1, 2)
# RPN anchors start at this size on RPN_MIN_LEVEL
# The anchor size doubled each level after that
# With a default of 32 and levels 2 to 6, we get anchor sizes of 32 to 512
__C.FPN.RPN_ANCHOR_START_SIZE = 32
__C.FPN.EXTRA_CONV_LEVELS = False
# Compatibility stopgap measure for some experimental models
__C.FPN.INPLACE_LATERAL = False


# ---------------------------------------------------------------------------- #
# Mask R-CNN options
# ---------------------------------------------------------------------------- #
__C.MRCNN = AttrDict()

__C.MRCNN.MASK_HEAD_NAME = b''
# Resolution of mask predictions
__C.MRCNN.RESOLUTION = 14  # TODO: rename to MASK_RESOLUTION
__C.MRCNN.ROI_XFORM_METHOD = b'RoIAlign'
__C.MRCNN.ROI_XFORM_RESOLUTION = 7
__C.MRCNN.ROI_XFORM_SAMPLING_RATIO = 0
__C.MRCNN.DIM_REDUCED = 256
__C.MRCNN.THRESH_BINARIZE = 0.5
__C.MRCNN.WEIGHT_LOSS_MASK = 1.
__C.MRCNN.CLS_SPECIFIC_MASK = True
__C.MRCNN.DILATION = 2  # TODO(rbg): not supported in ResNet conv5 head yet
__C.MRCNN.UPSAMPLE_RATIO = 1
__C.MRCNN.USE_FC_OUTPUT = False
__C.MRCNN.CONV_INIT = b'GaussianFill'


# ---------------------------------------------------------------------------- #
# Keyoint R-CNN options
# ---------------------------------------------------------------------------- #
__C.KRCNN = AttrDict()

# Keypoint prediction head options
__C.KRCNN.ROI_KEYPOINTS_HEAD = b''
# Output size (and size loss is computed on), e.g., 56x56
__C.KRCNN.HEATMAP_SIZE = -1
__C.KRCNN.UP_SCALE = -1
__C.KRCNN.USE_DECONV = False
__C.KRCNN.USE_DECONV_OUTPUT = False
__C.KRCNN.DILATION = 1
__C.KRCNN.DECONV_KERNEL = 4
__C.KRCNN.DECONV_DIM = 256
__C.KRCNN.NUM_KEYPOINTS = -1
__C.KRCNN.CONV_HEAD_DIM = 256
__C.KRCNN.CONV_HEAD_KERNEL = 3
__C.KRCNN.CONV_INIT = b'GaussianFill'
# Use NMS based on OKS
__C.KRCNN.NMS_OKS = False
# Source for keypoint confidence
#   Valid options: ('bbox', 'logit', 'prob')
__C.KRCNN.KEYPOINT_CONFIDENCE = b'bbox'
# Standard ROI XFORM options
__C.KRCNN.ROI_XFORM_METHOD = b'RoIAlign'
__C.KRCNN.ROI_XFORM_RESOLUTION = 7
__C.KRCNN.ROI_XFORM_SAMPLING_RATIO = 0
# Minimum number of labeled keypoints that must exist in a minibatch (otherwise
# the minibatch is discarded)
__C.KRCNN.MIN_KEYPOINT_COUNT_FOR_VALID_MINIBATCH = 20
__C.KRCNN.NUM_STACKED_CONVS = 8
__C.KRCNN.INFERENCE_MIN_SIZE = 0
__C.KRCNN.LOSS_WEIGHT = 1.0
# Use 3D deconv for videos
# Setting true will only activate for video inputs though
__C.KRCNN.USE_3D_DECONV = False
# Set to False if you want to move time to channel dim when computing the keypoint
# outputs. Set to True and it will move to batch dimension.
__C.KRCNN.NO_3D_DECONV_TIME_TO_CH = False

# ---------------------------------------------------------------------------- #
# VIDEO
# ---------------------------------------------------------------------------- #
__C.VIDEO = AttrDict()

__C.VIDEO.NUM_FRAMES = -1
# The temporal dimension at the ROIalign stage. By default will set to same as
# NUM_FRAMES (assuming no temporal strides)
__C.VIDEO.NUM_FRAMES_MID = -1
__C.VIDEO.TIME_INTERVAL = -1
# Could be 'center-only', 'mean-repeat' etc (see lib/utils/net.py)
__C.VIDEO.WEIGHTS_INFLATE_MODE = b''

# Time kernel dims, for each part of the network
__C.VIDEO.TIME_KERNEL_DIM = AttrDict()
__C.VIDEO.TIME_KERNEL_DIM.BODY = 1
__C.VIDEO.TIME_KERNEL_DIM.HEAD_RPN = 1
__C.VIDEO.TIME_KERNEL_DIM.HEAD_KPS = 1
__C.VIDEO.TIME_KERNEL_DIM.HEAD_DET = 1  # only used for ResNet heads (FPN uses MLP)

# Set to True, it will use the same stride as on the spatial dimensions
__C.VIDEO.TIME_STRIDE_ON = False
# Set this to 'avg', 'slice-center' or '' (do nothing, which requires a 3D head)
__C.VIDEO.BODY_HEAD_LINK = b''
# Predict "vis" labels for tube. This is to take care of the case when all the
# boxes predicted are not in the track
__C.VIDEO.PREDICT_RPN_BOX_VIS = False

# Set to true if you want to use the GT values of RPN. This basically avoids
# evaluating the RPN in training
__C.VIDEO.DEBUG_USE_RPN_GT = False
# How to generate TPN.
# replicate = copy over the t=1 proposal num_frame times
# combinations = compute all possible combinations
__C.VIDEO.RPN_TUBE_GEN_STYLE = b'replicate'
# Default frames/clips to extract from video datasets for training/testing.
# IF the datasets/json_dataset.py entry has a different number, that will take
# precedence over this.
__C.VIDEO.DEFAULT_CLIPS_PER_VIDEO = 9999999999  # default, take all


# ---------------------------------------------------------------------------- #
# External Paths (to open source code, etc)
# ---------------------------------------------------------------------------- #
__C.EXT_PATHS = AttrDict()
# This is an old version of code
# https://github.com/leonid-pishchulin/poseval/tree/39fd82bc328b3b6d580c7afe2e98316cba35ab4a  # noQA
# __C.EXT_PATHS.POSEVAL_CODE_PATH = b'/home/rgirdhar/local/OpenSource/github/poseval/'
# Multi-processing version of
# https://github.com/leonid-pishchulin/poseval/tree/dfb11f7c1035ae7d91f1601fdd1972897c2a7cf4
__C.EXT_PATHS.POSEVAL_CODE_PATH = b'/home/rgirdhar/local/OpenSource/bitbucket/poseval/'


# ---------------------------------------------------------------------------- #
# ResNets options (ResNet/ResNeXt)
# ---------------------------------------------------------------------------- #
__C.RESNETS = AttrDict()
# by default, we support the MSRA ResNet50
__C.RESNETS.NUM_GROUPS = 1
__C.RESNETS.WIDTH_PER_GROUP = 64
__C.RESNETS.STRIDE_1X1 = True  # True only for MSRA ResNet; False for C2/Torch
__C.RESNETS.TRANS_FUNC = b'bottleneck_transformation'


# ---------------------------------------------------------------------------- #
# Tracking parameters
# ---------------------------------------------------------------------------- #
__C.TRACKING = AttrDict()
# Confidence value of detections to keep. Drop the lower conf detections before
# running tracking.
__C.TRACKING.CONF_FILTER_INITIAL_DETS = 0.9
# Set the following if you want to run tracking on a specific detections file.
# By default it will pick the one in the test directory corresponding to the
# config file
__C.TRACKING.DETECTIONS_FILE = b''
# Tracking distance metrics
__C.TRACKING.DISTANCE_METRICS = ('bbox-overlap', 'cnn-cosdist', 'pose-pck')
__C.TRACKING.DISTANCE_METRIC_WTS = (1.0, 0.0, 0.0)
# Algorithm to use for matching between frames
__C.TRACKING.BIPARTITE_MATCHING_ALGO = b'hungarian'
# Layer to use for CNN feature based matching for tracking, w.r.t resnet18 in pytorch
__C.TRACKING.CNN_MATCHING_LAYER = b'layer3'
# Pose smoothing
__C.TRACKING.FLOW_SMOOTHING_ON = False
# How to set the conf for each keypoint. ['global'/'local'/'scaled']
__C.TRACKING.KP_CONF_TYPE = b'global'

# Flow smoothing
__C.TRACKING.FLOW_SMOOTHING = AttrDict()
# When it is a scene change
__C.TRACKING.FLOW_SMOOTHING.FLOW_SHOT_BOUNDARY_TH = 6.0
# How many frames to consider
__C.TRACKING.FLOW_SMOOTHING.N_CONTEXT_FRAMES = 3
# Extend tracks to frames which do not have that track. Else it will only
# smooth the poses that already existed in that frame
__C.TRACKING.FLOW_SMOOTHING.EXTEND_TRACKS = True

# Keep center detections only, and drop the other frames (even before tracking).
# This basically reduces a 3D model output back to 2D by only keeping predictions
# corresponding to the center frame
# Keeping it true as I'm not doing any tube-level tracking so far. Once everything
# else works, implement that.
__C.TRACKING.KEEP_CENTER_DETS_ONLY = True

# debug
__C.TRACKING.DEBUG = AttrDict()
# Set the following to get labels from the GT for this frame. This avoids
# tracks from getting lost
__C.TRACKING.DEBUG.UPPER_BOUND = False
# Set the following if you also want to copy the keypoint locations from the GT.
# This gives an idea if all the kps for detected boxes were correct, what num
# will I get. i.e., if only issue was missed boxes
__C.TRACKING.DEBUG.UPPER_BOUND_2_GT_KPS = False
# Set the following to not copy the keypoints, only the conf value
__C.TRACKING.DEBUG.UPPER_BOUND_2_GT_KPS_ONLY_CONF = False
# This ensures the shot boundaries are known
__C.TRACKING.DEBUG.UPPER_BOUND_3_SHOTS = False
# This uses upper bound in the evaluation code, copying over the GT track id
# to the detection
__C.TRACKING.DEBUG.UPPER_BOUND_4_EVAL_UPPER_BOUND = False
# To evaluate if I only replaced the keypoints, without replacing the track Ids from GT
__C.TRACKING.DEBUG.UPPER_BOUND_5_GT_KPS_ONLY = False
# Debugging flow smoothing
__C.TRACKING.DEBUG.FLOW_SMOOTHING_COMBINE = False
# The dummy tracks baseline
__C.TRACKING.DEBUG.DUMMY_TRACKS = False

# Training a LSTM for tracking
__C.TRACKING.LSTM = AttrDict()
# type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)
__C.TRACKING.LSTM.MODEL = b'LSTM'
# Initial Language embedding size
__C.TRACKING.LSTM.EMSIZE = 200
# Number of hidden units per layer
__C.TRACKING.LSTM.NHID = 200
# Number of layers
__C.TRACKING.LSTM.NLAYERS = 2
__C.TRACKING.LSTM.DROPOUT = 0.2
# Tie the weights of the encoder and decoder (only works if the hidden dim ==
# encoded dim)
__C.TRACKING.LSTM.TIED_WTS = False
# Initial LR for the LSTM
__C.TRACKING.LSTM.LR = 0.1
# Gradient clipping for the LSTM
__C.TRACKING.LSTM.GRAD_CLIP = 0.25
__C.TRACKING.LSTM.BATCH_SIZE = 20
__C.TRACKING.LSTM.EPOCHS = 10
__C.TRACKING.LSTM.LOG_INTERVAL = 200
# If True, it will incur loss only on the last prediction, and not on the
# intermediate predictions
__C.TRACKING.LSTM.LOSS_LAST_PRED_ONLY = False
# Features to consider for matching using LSTM. Options 'bbox'/'kpts'
__C.TRACKING.LSTM.FEATS_TO_CONSIDER = ['bbox', 'kpts']
__C.TRACKING.LSTM.NUM_WORKERS = 4
# Consider small length tracks too. Normally it will take only the NUM_FRAMES
# length tracks and consider subsets of that.
__C.TRACKING.LSTM.CONSIDER_SHORT_TRACKS_TOO = False

# Tracking test time config
__C.TRACKING.LSTM_TEST = AttrDict()
__C.TRACKING.LSTM_TEST.LSTM_TRACKING_ON = False
__C.TRACKING.LSTM_TEST.LSTM_WEIGHTS = b''  # The path to LSTM .pt file

# ---------------------------------------------------------------------------- #
# Evaluation parameters
# ---------------------------------------------------------------------------- #
__C.EVAL = AttrDict()
# Run the per-video evaluation in eval_mpii.py. This is used to rank vis by the
# score received on a specific video
__C.EVAL.EVAL_MPII_PER_VIDEO = False
# Drop all detections below this threshold when evaluating. The 0.5 default
# exists as that was set by @gkioxari ealier in the earlier version, but ideally
# should be set to 0, as in mAP evaluation there is no reason to drop detections.
# In tracking, the compute_tracks.py code will itself drop the extra
# detections, so no need to worry about it in evaluation code.
__C.EVAL.EVAL_MPII_DROP_DETECTION_THRESHOLD = 0.5
# Set a threshold on what keypoints end up into the final evaluation. By
# default I put in every single keypoint
__C.EVAL.EVAL_MPII_KPT_THRESHOLD = -float('Inf')

# ---------------------------------------------------------------------------- #
# Misc options
# ---------------------------------------------------------------------------- #
# Number of GPUs to use
__C.NUM_GPUS = 1

# Use NCCL for all reduce, otherwise use muji
# NCCL seems to work ok for 2 GPUs, but become prone to deadlocks when using
# 4 or 8
__C.USE_NCCL = False

# The mapping from image coordinates to feature map coordinates might cause
# some boxes that are distinct in image space to become identical in feature
# coordinates. If DEDUP_BOXES > 0, then DEDUP_BOXES is used as the scale factor
# for identifying duplicate boxes.
# 1/16 is correct for {Alex,Caffe}Net, VGG_CNN_M_1024, and VGG16
__C.DEDUP_BOXES = 1 / 16.

__C.BBOX_XFORM_CLIP = np.log(1000. / 16.)

# Pixel mean values (BGR order) as a (1, 1, 3) array
# We use the same pixel mean for all networks even though it's not exactly what
# they were trained with
__C.PIXEL_MEANS = np.array([[[102.9801, 115.9465, 122.7717]]])

# For reproducibility
__C.RNG_SEED = 3

# A small number that's used many times
__C.EPS = 1e-14

# Root directory of project
# __C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
__C.ROOT_DIR = os.getcwd()

# Output basedir
__C.OUTPUT_DIR = b'/tmp'

# Name (or path to) the matlab executable
__C.MATLAB = b'matlab'

# Directory where VOCdevkit<year> directories can be found
__C.VOC_DIR = b'/mnt/vol/gfsai-east/ai-group/datasets'

# Root GPU device id (always leave at 0; use CUDA_VISIBLE_DEVICES env var)
__C.ROOT_GPU_ID = 0

# Reduce memory usage with memonger gradient blob sharing
__C.MEMONGER = True
# Futher reduce memory by allowing forward pass activations to be shared when
# possible. Note that this will cause activation blob inspection (values,
# shapes, etc.) to be meaningless when activation blobs are reused.
__C.MEMONGER_SHARE_ACTIVATIONS = False

# Dump detection visualizations
__C.VIS = False
# if we do ensembling, we need a smaller threshold to visualize
__C.VIS_THR = 0.9

# A final message to print at the end of training (e.g., this can be used to
# print expected results for a test config to make visual comparison easy)
__C.FINAL_MSG = b''

# Use a subset of initial roidb. Useful for debugging and running test on
# huge datasets across nodes
__C.ROIDB_SUBSET = []

# Number of processes to read data
__C.NUM_WORKERS = 4


# ---------------------------------------------------------------------------- #
# Facebook cluster specific options (not needed in OSS code)
# ---------------------------------------------------------------------------- #
__C.CLUSTER = AttrDict()
__C.CLUSTER.ON_CLUSTER = False
__C.CLUSTER.AUTO_RESUME = True


# ---------------------------------------------------------------------------- #
# Devstorage node mounting options
# ---------------------------------------------------------------------------- #
__C.DEVSTORAGE = AttrDict()
__C.DEVSTORAGE.MOUNT_ENABLED = False
__C.DEVSTORAGE.HOSTNAME = b''
__C.DEVSTORAGE.REMOTE_PATH = b''
__C.DEVSTORAGE.MOUNT_POINT = b'/tmp/devstorage'


# ---------------------------------------------------------------------------- #
# Debug options
# ---------------------------------------------------------------------------- #
__C.DEBUG = AttrDict()
__C.DEBUG.DATA_LOADING = False
__C.DEBUG.STOP_TRAIN_ITER = False

# ---------------------------------------------------------------------------- #
# Deprecated options
# If an option is removed from the code and you don't want to break existing
# yaml configs, you can add the config key with the suffix '_deprecated' below
# ---------------------------------------------------------------------------- #
__C.TRAIN.BBOX_NORMALIZE_TARGETS_PRECOMPUTED_deprecated = None
__C.USE_GPU_NMS_deprecated = None


# Consumers can get the config with default values by:
#   from core.config import cfg_default
cfg_default = copy.deepcopy(__C)


def assert_and_infer_cfg():
    if __C.MODEL.RPN_ONLY or __C.MODEL.FASTER_RCNN:
        __C.RPN.ON = True
    if __C.MODEL.RPN_ONLY:
        # RPN-only models do not have Fast R-CNN style bbox regressors
        __C.TRAIN.BBOX_REG = False
    # Video inferences
    if __C.VIDEO.NUM_FRAMES_MID == -1:
        __C.VIDEO.NUM_FRAMES_MID = __C.VIDEO.NUM_FRAMES
    # BN test only spec if SpatialBN is used
    assert (not __C.MODEL.USE_BN_TESTMODE_ONLY) or __C.MODEL.USE_BN


def get_output_dir(training=True):
    dataset = __C.TRAIN.DATASET if training else __C.TEST.DATASET
    tag = 'train' if training else 'test'
    # <output-dir>/<train|test>/<dataset>/<model-type>/
    outdir = osp.join(__C.OUTPUT_DIR, tag, dataset, __C.MODEL.TYPE)
    if not osp.exists(outdir):
        os.makedirs(outdir)
    return outdir


def _merge_a_into_b(a, b):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    from ast import literal_eval
    if not isinstance(a, AttrDict):
        return

    for k, v in a.items():
        # a must specify keys that are in b
        if k not in b:
            if k + '_deprecated' in b:
                logger.warn('Config key {} is deprecated, ignoring'.format(k))
                return
            else:
                raise KeyError('{} is not a valid config key'.format(k))

        if type(v) is dict:
            a[k] = v = AttrDict(v)
        if isinstance(v, basestring):  # NoQA
            try:
                v = literal_eval(v)
            except BaseException:
                pass

        # the types must match, too (with some exceptions)
        old_type = type(b[k])
        if old_type is not type(v) and v is not None:
            if isinstance(b[k], np.ndarray):
                v = np.array(v, dtype=b[k].dtype)
            elif isinstance(b[k], basestring) and isinstance(v, unicode):  # NoQA
                v = str(v)
            else:
                raise ValueError(
                    'Type mismatch ({} vs. {}) for config key: {}'.format(
                        type(b[k]), type(v), k))

        # recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                _merge_a_into_b(a[k], b[k])
            except BaseException:
                logger.critical('Error under config key: {}'.format(k))
                raise
        else:
            b[k] = v


############################
### All config fix rules ###
############################

def _fix_video_time_kernel_dim(a):
    # VIDEO.TIME_KERNEL_DIM  # no longer allowed, map to more specific values
    if ('VIDEO' in a) and ('TIME_KERNEL_DIM' in a['VIDEO']) and \
            isinstance(a['VIDEO']['TIME_KERNEL_DIM'], int):
        val = a['VIDEO']['TIME_KERNEL_DIM']
    else:
        # Didn't exist in a, or was already fixed
        return a
    a['VIDEO']['TIME_KERNEL_DIM'] = {}
    for sub_class in __C.VIDEO.TIME_KERNEL_DIM.keys():
        a['VIDEO']['TIME_KERNEL_DIM'][sub_class] = val
    return a

#################################
### End: All config fix rules ###
#################################


def _config_mapping_rules(a):
    """ a is a config dictionary (AttrDict). Here I define clean up rules,
    especially for older config options that are now deprecated and can be
    mapped to newer config options. """
    a = _fix_video_time_kernel_dim(a)
    return a


def cfg_from_file(filename):
    """Load a config file and merge it into the default options."""
    import yaml
    with open(filename, 'r') as f:
        yaml_cfg = yaml.load(f)
        yaml_cfg = _config_mapping_rules(yaml_cfg)
        yaml_cfg = AttrDict(yaml_cfg)

    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_cfg(yaml_cfg):
    _merge_a_into_b(yaml_cfg, __C)


def cfg_from_list(cfg_list):
    """Set config keys via list (e.g., from command line)."""
    from ast import literal_eval
    assert len(cfg_list) % 2 == 0
    for k, v in zip(cfg_list[0::2], cfg_list[1::2]):
        key_list = k.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d, 'Config key {} not found'.format(subkey)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'Config key {} not found'.format(subkey)
        try:
            value = literal_eval(v)
        except BaseException:
            # handle the case when v is a string literal
            value = v
        assert isinstance(value, type(d[subkey])) or d[subkey] is None, \
            'type {} does not match original type {}'.format(
                type(value), type(d[subkey]))
        d[subkey] = value
