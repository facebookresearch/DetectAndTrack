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
import warnings
import errno
import functools
import subprocess

import logging
logger = logging.getLogger(__name__)


def mkdir_p(path):
    """
    Make all directories in `path`. Ignore errors if a directory exists.
    Equivalent to `mkdir -p` in the command line, hence the name.
    """
    try:
        os.makedirs(path)
    except OSError as exc:  # Python >2.5
        if exc.errno == errno.EEXIST and os.path.isdir(path):
            pass
        else:
            raise


def get_best_accessible(paths):
    """If paths is a list of paths, then return the first one that is
    accessible from this script. This is useful when I want to run stuff
    with a different path locally than on the cluster.
    """
    if isinstance(paths, list):
        for path in paths:
            if os.path.exists(path):
                return path
        logger.error('None of the paths {} were accessible!'.format(paths))
        # as the last one is probably the slowest => most accessible path
        return paths[-1]
    else:
        return paths  # could be str/unicode something...


def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emmitted
    when the function is used."""

    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning, stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func


def static_vars(**kwargs):
    """ Decorator for static vars.
    From https://stackoverflow.com/a/279586/1492614. """
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func
    return decorate


def flatten_list(l):
    # Thanks to https://stackoverflow.com/a/952952
    return [item for sublist in l for item in sublist]


def run_cmd(cmd, print_cmd=True, return_output=False, user=None):
    """
    Returns true if successful, else false.
    If return_output == True, then instead returns the stdout output.
    """
    cmd = cmd.strip()
    if user is not None:
        cmd = 'sudo runuser -l {} -c \'{}\''.format(user, cmd)
    if print_cmd:
        # This only works with the local anaconda installation
        from termcolor import colored
        print('Running', colored('{}'.format(cmd), 'green'))
    # older way...
    # print('Running \033[0;32m {} \033[0;0m'.format(cmd))
    if return_output:
        return subprocess.check_output(cmd, shell=True).strip() # NoQA (ignored shlex.quote for now)
    else:
        return subprocess.call(cmd, shell=True) == 0 # NoQA (ignored shlex.quote for now)
