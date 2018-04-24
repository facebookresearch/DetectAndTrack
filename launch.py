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
import subprocess


def parse_args():
    parser = argparse.ArgumentParser(description='Launch a config')
    parser.add_argument(
        '--cfg', '-c', dest='cfg_file', required=True,
        help='Config file to run')
    parser.add_argument(
        '--mode', '-m', dest='mode',
        help='Mode to run [train/test/track/eval]',
        default='train')
    parser.add_argument(
        'opts', help='See lib/core/config.py for all options', default=None,
        nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def _run_cmd(tool_file, cfg_file, other_opts):
    cmd = '''python tools/{tool_file} \
             --cfg {cfg_file} \
             {other_opts}
          '''.format(tool_file=tool_file,
                     cfg_file=cfg_file,
                     other_opts=other_opts)
    subprocess.call(cmd, shell=True)


def main():
    args = parse_args()
    other_opts = ''
    if args.mode in ['train', 'test']:
        other_opts += '--multi-gpu-testing '
    other_opts += 'OUTPUT_DIR outputs/{}  '.format(
        args.cfg_file)
    if args.opts is not None:
        other_opts += ' '.join(args.opts)
    tool_file = 'train_net.py'
    if args.mode == 'test':
        tool_file = 'test_net.py'
    elif args.mode == 'track':
        tool_file = 'compute_tracks.py'
    elif args.mode == 'eval':
        tool_file = 'eval_mpii.py'
    _run_cmd(tool_file, args.cfg_file, other_opts)


if __name__ == '__main__':
    main()
