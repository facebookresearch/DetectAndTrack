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

import numpy as np
from collections import deque
import json
# Print lower precision floating point values than default FLOAT_REPR
json.encoder.FLOAT_REPR = lambda o: format(o, '.6f')


def log_json_stats(stats, json_out_file=None, sort_keys=True):
    json_str = json.dumps(stats, sort_keys=sort_keys)
    print('json_stats: {:s}'.format(json_str))
    if json_out_file is not None:
        with open(json_out_file, 'a') as fout:
            fout.write('{:s}\n'.format(json_str))


class SmoothedValue(object):
    """Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size):
        self.deque = deque(maxlen=window_size)
        self.series = []
        self.total = 0.0
        self.count = 0

    def AddValue(self, value):
        self.deque.append(value)
        self.series.append(value)
        self.count += 1
        self.total += value

    def GetMedianValue(self):
        return np.median(self.deque)

    def GetAverageValue(self):
        return np.mean(self.deque)

    def GetGlobalAverageValue(self):
        return self.total / self.count
