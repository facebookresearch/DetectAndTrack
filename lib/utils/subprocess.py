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
import yaml
import numpy as np
import subprocess
import cPickle as pickle
from six.moves import shlex_quote

from core.config import cfg

import logging
logger = logging.getLogger(__name__)


def process_in_parallel(tag, size, binary, output_dir, opt_args=None):
    """Runs binary NUM_GPUS times in parallel, each using one GPU. Assumes the
    binary implements command line arguments `--range {start} {end}`.
    """
    # Snapshot the current cfg state in order to pass to the inference
    # subprocesses
    cfg_file = os.path.join(output_dir, '{}_range_config.yaml'.format(tag))
    with open(cfg_file, 'w') as f:
        yaml.dump(cfg, stream=f)
    subprocess_env = os.environ.copy()
    processes = []
    subinds = np.array_split(range(size), cfg.NUM_GPUS)
    for i in range(cfg.NUM_GPUS):
        start = subinds[i][0]
        end = subinds[i][-1] + 1
        subprocess_env['CUDA_VISIBLE_DEVICES'] = str(i)
        cmd = '{binary} --range {start} {end} --cfg {cfg_file}'
        cmd = cmd.format(
            binary=shlex_quote(binary),
            start=int(start),
            end=int(end),
            cfg_file=shlex_quote(cfg_file))
        if opt_args is not None:
            for arg_name, arg_val in opt_args:
                cmd = '{} --{} {}'.format(cmd, arg_name, arg_val)
        cmd = '{} NUM_GPUS 1'.format(cmd)
        logger.info('{} range command {}: {}'.format(tag, i, cmd))
        if i == 0:
            subprocess_stdout = subprocess.PIPE
        else:
            filename = os.path.join(
                output_dir, '%s_range_%s_%s.stdout' % (tag, start, end))
            subprocess_stdout = open(filename, 'w')  # NOQA (close below)
        p = subprocess.Popen(
            cmd, shell=True, env=subprocess_env, stdout=subprocess_stdout,
            stderr=subprocess.STDOUT, bufsize=1)
        processes.append((i, p, start, end, subprocess_stdout))
    # Log output from inference processes and collate their results
    outputs = []
    for i, p, start, end, subprocess_stdout in processes:
        log_subprocess_output(i, p, output_dir, tag, start, end)
        if isinstance(subprocess_stdout, file):  # NOQA (Python 2 for now)
            subprocess_stdout.close()
        range_file = os.path.join(
            output_dir, '%s_range_%s_%s.pkl' % (tag, start, end))
        range_data = pickle.load(open(range_file))
        outputs.append(range_data)
    return outputs


def log_subprocess_output(i, p, output_dir, tag, start, end):
    outfile = os.path.join(
        output_dir, '%s_range_%s_%s.stdout' % (tag, start, end))
    logger.info('# ' + '-' * 76 + ' #')
    logger.info(
        'stdout of subprocess %s with range [%s, %s]' % (i, start + 1, end))
    logger.info('# ' + '-' * 76 + ' #')
    if i == 0:
        # Stream the piped stdout from the first subprocess in realtime
        with open(outfile, 'w') as f:
            for line in iter(p.stdout.readline, b''):
                print(line.rstrip())
                f.write(str(line))
        p.stdout.close()
        ret = p.wait()
    else:
        # For subprocesses >= 1, wait and dump their log file
        ret = p.wait()
        with open(outfile, 'r') as f:
            print(''.join(f.readlines()))
    assert ret == 0, 'Range subprocess failed (exit code: {})'.format(ret)
