##############################################################
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################


"""Test a Fast R-CNN network on an image database."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)

from caffe2.python import workspace
from core.rpn_generator import generate_rpn_on_range, generate_rpn_on_dataset
from core.config import (cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg,
                         get_output_dir)
import utils.c2

import argparse
import pprint
import time
import os
import sys
import logging
import numpy as np

utils.c2.import_detectron_ops()

FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description='Test a Fast R-CNN network')
    parser.add_argument(
        '--cfg', dest='cfg_file', help='optional config file', default=None,
        type=str)
    parser.add_argument(
        '--wait', dest='wait', help='wait until net file exists', default=True,
        type=bool)
    parser.add_argument(
        '--vis', dest='vis', help='visualize detections', action='store_true')
    parser.add_argument(
        '--multi-gpu-testing', dest='multi_gpu_testing',
        help='using cfg.NUM_GPUS for inference', action='store_true')
    parser.add_argument(
        '--range', dest='range',
        help='start (inclusive) and end (exclusive) indices',
        default=None, type=int, nargs=2)
    parser.add_argument(
        'opts', help='See lib/core/config.py for all options', default=None,
        nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main(ind_range=None, multi_gpu_testing=False):
    if cfg.TEST.EXT_CNN_FEATURES:
        import core.feat_engine as engine
    else:
        import core.test_engine as engine
    if cfg.MODEL.RPN_ONLY:
        if ind_range is not None:
            # Subprocess child case:
            #
            # In this case test_net was called via subprocess.Popen to execute on a
            # range of inputs on a single dataset (i.e., use cfg.TEST.DATASET and
            # don't loop over cfg.TEST.DATASETS)
            generate_rpn_on_range(ind_range=ind_range)
        else:
            # Parent case:
            #
            # In this case we're either running inference on the entire dataset in a
            # single process or using this process to launch subprocesses that each
            # run inference on a range of the dataset
            if len(cfg.TEST.DATASETS) == 0:
                cfg.TEST.DATASETS = (cfg.TEST.DATASET, )

            for i in range(len(cfg.TEST.DATASETS)):
                cfg.TEST.DATASET = cfg.TEST.DATASETS[i]
                generate_rpn_on_dataset(multi_gpu=multi_gpu_testing)
    else:
        if ind_range is not None:
            # Child (see comment above)
            engine.test_net(ind_range=ind_range)
        else:
            # Parent (see comment above)
            if len(cfg.TEST.DATASETS) == 0:
                cfg.TEST.DATASETS = (cfg.TEST.DATASET, )
                cfg.TEST.PROPOSAL_FILES = (cfg.TEST.PROPOSAL_FILE, )

            for i in range(len(cfg.TEST.DATASETS)):
                cfg.TEST.DATASET = cfg.TEST.DATASETS[i]
                cfg.TEST.PROPOSAL_FILE = cfg.TEST.PROPOSAL_FILES[i]
                engine.test_net_on_dataset(multi_gpu=multi_gpu_testing)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=0'])
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.opts is not None:
        cfg_from_list(args.opts)
    assert_and_infer_cfg()
    logger.info('Testing with config:')
    logger.info(pprint.pformat(cfg))

    if not os.path.exists(cfg.TEST.WEIGHTS):
        # Check if there's a trained model stored in this directory
        output_dir = get_output_dir(training=True)
        train_ckpt_path = os.path.join(output_dir, 'model_final.pkl')
        if os.path.exists(train_ckpt_path):
            cfg.TEST.WEIGHTS = train_ckpt_path
        else:
            # Take the longest trained model so far
            potential_ckpts = [pth for pth in os.listdir(output_dir)
                               if pth.startswith('model_iter')]
            potential_ckpts_nums = [int(el[len('model_iter'):-len('.pkl')]) for
                                    el in potential_ckpts]
            if len(potential_ckpts_nums) > 0:
                cfg.TEST.WEIGHTS = os.path.join(
                    output_dir,
                    potential_ckpts[np.argmax(potential_ckpts_nums)])
        logger.info('No test weights specified but found the trained '
                    'model here {}. Using that for testing.'.format(
                        cfg.TEST.WEIGHTS))
    while not os.path.exists(cfg.TEST.WEIGHTS) and args.wait:
        logger.info('Waiting for {} to exist...'.format(cfg.TEST.WEIGHTS))
        time.sleep(10)

    main(ind_range=args.range, multi_gpu_testing=args.multi_gpu_testing)
