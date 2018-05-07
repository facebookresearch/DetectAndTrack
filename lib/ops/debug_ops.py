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
import pdb
import traceback as tb
import code


class PDBOp(object):
    """ Simply for debugging. Do a pdb in the graph.
    """

    def forward(self, input_tensor, output_tensor):
        inputs = [el.data for el in input_tensor]
        # some standard debugging stuff
        print('>> Inputs have NaN?: {}'.format([
            np.any(np.isnan(el)) for el in inputs]))
        pdb.set_trace()


class TDBOp(object):
    """ Better version of PDB
    """

    def forward(self, input_tensor, output_tensor):
        inputs = [el.data for el in input_tensor]
        # some standard debugging stuff
        print('>> Inputs have NaN?: {}'.format([
            np.any(np.isnan(el)) for el in inputs]))
        tb.print_stack()
        namespace = globals().copy()
        namespace.update(locals())
        code.interact(local=namespace)
