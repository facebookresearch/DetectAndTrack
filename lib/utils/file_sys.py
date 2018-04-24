##############################################################
# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
##############################################################

"""Provides utility functions for interacting with the file system."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import errno
import os
import subprocess
import time


def mkdir_exists(path):
    """Makes a directory if it does not exist already."""
    try:
        os.mkdir(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise


def makedirs_exists(path):
    """Like mkdir_exists but makes all intermediate-level dirs."""
    try:
        os.makedirs(path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise


def run_command_with_retries(args, num_retries=4):
    """Runs a given command with retries."""
    ret_code = 1
    for retry in range(num_retries):
        ret_code = subprocess.call(args)
        if ret_code == 0:
            break
        time.sleep(2**retry)
    assert ret_code == 0, \
        'Return code from \'{}\' was {}'.format(' '.join(args), ret_code)


def rsync(src, dest):
    """Performs rsync with retries."""
    run_command_with_retries(['rsync', src, dest])


def nfusr(hostname, remote_path, mount_point):
    """Performs userspace mount."""
    run_command_with_retries([
        'nfusr', 'nfs://{0}{1}'.format(hostname, remote_path), mount_point])


def umount_user(mount_point):
    """Unmounts a userspace mount."""
    run_command_with_retries(['fusermount', '-u', mount_point])
