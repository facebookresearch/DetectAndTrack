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
import cv2
import numpy as np
import logging
import sys

import torch
from torchvision import models
os.environ['TORCH_MODEL_ZOO'] = \
    '/mnt/vol/gfsai-east/ai-group/users/rgirdhar/StandardModels/PyTorch/ImNet'

FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)


default_model = models.resnet18(pretrained=True)


def prepare_image(im):
    im = im[..., (2, 1, 0)]  # convert to rgb
    try:
        im = cv2.resize(im, (224, 224))
    except cv2.error:
        im = np.zeros((224, 224, 3))  # dummy image
        logger.warning('Invalid patch, replaced with 0 image.')
    im = im.transpose(2, 0, 1)
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 3, 1, 1)
    std = np.array([0.229, 0.224, 0.224]).reshape(1, 3, 1, 1)
    im = (im / 255.0 - mean) / std
    im = torch.FloatTensor(im).cuda()
    im = torch.autograd.Variable(im, volatile=True)
    return im


def extract_features(im, test_model=None, layers=('layer3',)):
    """
    Args:
        im (np.ndarray): Image, read using cv2.imread so is in BGR format.
    Returns:
        features (list): List of features from each layer in the list layers.
    """
    model = test_model or default_model
    model.eval()
    # Preprocess the image
    im = prepare_image(im)

    # Extract the features
    x = im
    outputs = []
    layers = list(layers)
    for name, module in model._modules.items():
        if len(layers) == 0:
            break
        if name == 'fc':
            # Not sure why I need to do this...
            x = torch.squeeze(x)
        x = module.cuda()(x)
        if name in layers:
            outputs += [x.data.cpu().clone().numpy()]
            del layers[layers.index(name)]
    return outputs
