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

import argparse
import sys

from core.config import (
    cfg_from_file, assert_and_infer_cfg, get_output_dir, cfg_from_list)
from core.tracking_engine import run_posetrack_tracking
from core.test_engine import get_roidb_and_dataset


def _parse_args():
    parser = argparse.ArgumentParser(description='Perform matching.')
    parser.add_argument(
        '--cfg', dest='cfg_file',
        help='Config file',
        required=True,
        type=str)
    parser.add_argument(
        'opts', help='See lib/core/config.py for all options', default=None,
        nargs=argparse.REMAINDER)

    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def main():
    args = _parse_args()
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.opts is not None:
        cfg_from_list(args.opts)
    assert_and_infer_cfg()
    test_output_dir = get_output_dir(training=False)
    json_data, _, _, _, _ = get_roidb_and_dataset(None, include_gt=True)
    run_posetrack_tracking(test_output_dir, json_data)


if __name__ == '__main__':
    main()
