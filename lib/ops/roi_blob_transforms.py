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


class RoIToBatchFormatOp(object):
    """ Given a Nx(4T+1) blob, convert to N*Tx(4+1) blob to be used with
    RoITransform operations.
    TODO(rgirdhar): Also use the "visibility" predictions here and replace
    those RoIs with the full image. This would take care of handling those
    boxes in RoIAlign step (anyway the predictions for those won't incur loss).
    """

    def forward(self, input_tensor, output_tensor):
        input = input_tensor[0].data
        assert input.ndim == 2, 'Input must be Nx(4T+1)'
        assert (input.shape[1] - 1) % 4 == 0, 'Input must be Nx(4T+1)'
        T = (input.shape[1] - 1) // 4
        N = input.shape[0]
        output = np.zeros((N * T, 4 + 1))
        for t in range(T):
            output[t::T, 0] = input[:, 0] * T + t
            output[t::T, 1:] = input[:, 1 + 4 * t: 1 + 4 * (t + 1)]
        output_tensor[0].reshape(output.shape)
        output_tensor[0].data[...] = output

    # No need of a backward pass as the RoIs are generated using the RPN model,
    # and using the "approx joint training (2)" style of training faster-rcnn,
    # we don't backprop from RoIAlign step back to RPN. Hence, this can simply
    # be considered as an extension of the RPN
