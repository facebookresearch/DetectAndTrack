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

import collections


ConvStageInfo = collections.namedtuple(
    'ConvStageInfo',
    ['blobs', 'dims', 'spatial_scales'])
