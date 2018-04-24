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

import cPickle as pickle
import os
# import cluster_utils.io

# from core.config import cfg


def robust_pickle_dump(data_dict, file_name):
    file_name = os.path.abspath(file_name)
    with open(file_name, 'wb') as f:
        pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)

    # This currently adds too much overhead; need to optimize.
    #
    # if cfg.CLUSTER.ON_CLUSTER:
    #     cluster_utils.io.learngpu_robust_pickle_dump(data_dict, file_name)
    # else:
    #     with open(file_name, 'wb') as f:
    #         pickle.dump(data_dict, f, pickle.HIGHEST_PROTOCOL)
