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
import sys

from core.config import (
    cfg_from_file, assert_and_infer_cfg, get_output_dir, cfg_from_list)
from core.mpii_eval_engine import run_mpii_eval
from core.test_engine import get_roidb_and_dataset


def _parse_args():
    parser = argparse.ArgumentParser(description='Evaluation')
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
    roidb, dataset, _, _, _ = get_roidb_and_dataset(None)
    run_mpii_eval(test_output_dir, roidb, dataset)


if __name__ == '__main__':
    main()
