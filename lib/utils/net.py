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
import numpy as np
import utils.blob
from collections import OrderedDict
import yaml
import cPickle as pickle
import pprint

from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, core, dyndep
from core.config import cfg
from utils.io import robust_pickle_dump

import logging
logger = logging.getLogger(__name__)


def import_custom_ops():
    """Import custom and contrib ops needed by this software."""
    # TODO(rbg): Consider moving into either model_builder or
    # DetectionModelHelper

    dyndep.InitOpsLibrary('@/caffe2/caffe2/contrib/nccl:nccl_ops')
    ops_path = '@/experimental/deeplearning/rgirdhar/VideoPose/lib/ops'
    ops_to_import = [
        'roi_pool_f_op',
        'ps_roi_pool_op',
        'l1_loss_op',
        'smooth_l1_loss_op',
        'affine_channel_op',
        'sigmoid_cross_entropy_loss_op',
        'upsample_nearest_op',
        'integral_image_op',
        'roi_pool_late_quantization_op',
        'roi_warp_op',
        'roi_warp_max_op',
        'spatial_narrow_as_op',
        'sample_as_op',
        'roi_align_op',
        'batch_permutation_op',
        'affine_channel_nd_op',
        # Doesn't work yet.. the CUDA version is not available and will not compile
        # 'conv_transpose_3d_op',
    ]
    for op in ops_to_import:
        dyndep.InitOpsLibrary(ops_path + ':' + op)


def initialize_from_weights_file(model, weights_file, broadcast=True):
    """Initialize a model from weights stored in a pickled dictionary. If
    multiple GPUs are used, the loaded weights are synchronized on all GPUs,
    unless 'broadcast' is False.
    """
    initialize_gpu_0_from_weights_file(model, weights_file)
    if broadcast:
        broadcast_parameters(model)


def inflate_weights_2d(pretrained_w, ws_blob, src_name, src_blobs):
    # Sometimes I want to inflate a 2D kernel where I predict T times the
    # output than before. So just repmating should be good.
    # Repeat all dimensions that can be, and see if it matches
    assert(pretrained_w.ndim == ws_blob.ndim)
    inflated_wt = pretrained_w.copy()
    for dim_i in range(ws_blob.ndim):
        if ws_blob.shape[dim_i] % pretrained_w.shape[dim_i] != 0:
            logger.info('Cant inflate {} ({}) for {} (dim {} not repmat-able)'
                        .format(src_name, pretrained_w.shape, ws_blob.shape, dim_i))
            return ws_blob
        t = ws_blob.shape[dim_i] // pretrained_w.shape[dim_i]
        inflated_wt = np.tile(
            inflated_wt, np.array(
                [1] * dim_i + [t] + [1] * (ws_blob.ndim - dim_i - 1)))
        # repeat and divide by t. Center init doesnot make much sense in this case
        inflated_wt = np.apply_along_axis(lambda x: x / t, dim_i, inflated_wt)
    logger.info('Inflating {} ({}) for {} by mean-rep mode on each dim'.format(
        src_name, pretrained_w.shape, ws_blob.shape))
    assert(np.all(inflated_wt.shape == ws_blob.shape))
    return inflated_wt


def inflate_weights(pretrained_w, ws_blob, src_name, src_blobs):
    """
    Try to inflate the weights to match the weight matrix
    """
    # rgirdhar: try to add extra dimension and see if that
    # matches (3D convolutions)
    # -3 is the time axis
    if ws_blob.ndim != 5:
        # This is not a conv 3D blob..
        if ws_blob.ndim == pretrained_w.ndim:
            # A 2D conv..
            return inflate_weights_2d(pretrained_w, ws_blob, src_name, src_blobs)
        else:
            logger.info('Not trying to inflate {}'.format(src_name))
            return ws_blob
    ncopies = float(ws_blob.shape[-3])
    inflated_w = np.repeat(np.expand_dims(
        pretrained_w, axis=-3), ncopies, axis=-3)
    if cfg.VIDEO.WEIGHTS_INFLATE_MODE == \
            'mean-repeat':
        # Scale down to keep the final values similar
        inflated_w = inflated_w / ncopies
    elif cfg.VIDEO.WEIGHTS_INFLATE_MODE == \
            'repeat':
        inflated_w = inflated_w
    elif cfg.VIDEO.WEIGHTS_INFLATE_MODE == \
            'center-only':
        # Set all other weights 0 but the center one.
        # No need to scale by 1/ncopies this time, as only
        # 1 copy is being added effectively.
        # This should work when ncopies = 1 as well
        inflated_w[..., :int(ncopies / 2), :, :] = 0
        inflated_w[..., int(ncopies / 2) + 1:, :, :] = 0
    elif cfg.VIDEO.WEIGHTS_INFLATE_MODE == \
            'center-only-rest-rand':
        # Set all other weights random but the center one.
        mu = 0
        sigma = 0.001
        inflated_w[..., :int(ncopies / 2), :, :] = sigma * np.random.randn(
            *inflated_w[..., :int(ncopies / 2), :, :].shape) + mu
        inflated_w[..., int(ncopies / 2) + 1:, :, :] = sigma * np.random.randn(
            *inflated_w[..., int(ncopies / 2) + 1:, :, :].shape) + mu
        # Need to divide this time, as the rest values are not 0
        inflated_w = inflated_w / ncopies
    else:
        raise ValueError('Invalid INFLATE_MODE: {}'.format(
            cfg.VIDEO.WEIGHTS_INFLATE_MODE))
    pretrained_w = inflated_w
    if ws_blob.shape == pretrained_w.shape:
        logger.warning(
            'Workspace blob {} ({}) loaded with pretrained '
            'wts {} after inflating the weights by {} mode.'
            .format(
                src_name, ws_blob.shape,
                src_blobs[src_name].shape,
                cfg.VIDEO.WEIGHTS_INFLATE_MODE))
    else:
        logger.error('Workspace blob {} with shape {} does '
                     'not match weights file shape {} '
                     '(even after inflating to {})'.format(
                         src_name,
                         ws_blob.shape,
                         src_blobs[src_name].shape,
                         pretrained_w.shape))
        # previously it would crash here, but maybe instead
        # of crashing, just keep training from scratch?
    return pretrained_w


def initialize_gpu_0_from_weights_file(model, weights_file):
    logger.info('Loading from: {}'.format(weights_file))
    is_first_init = 'trainedCOCO' in weights_file
    ws_blobs = workspace.Blobs()
    with open(weights_file, 'r') as f:
        src_blobs = pickle.load(f)
    if 'cfg' in src_blobs:
        saved_cfg = yaml.load(src_blobs['cfg'])
        configure_bbox_reg_weights(model, saved_cfg)
    if 'blobs' in src_blobs:
        # Backwards compat--dictionary used to be only blobs, now they are
        # stored under the 'blobs' key
        src_blobs = src_blobs['blobs']
    # Initialize weights on GPU 0 only
    unscoped_param_names = OrderedDict()  # Print these out in model order
    for blob in model.params:
        unscoped_param_names[utils.blob.unscope_name(str(blob))] = True
    with core.NameScope('gpu_0'):
        with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
            for unscoped_param_name in unscoped_param_names.keys():
                if (unscoped_param_name.find(']_') >= 0 and
                        unscoped_param_name not in src_blobs):
                    # Special case for sharing initialization from a pretrained
                    # model:
                    # If a blob named '_[xyz]_foo' is in model.params and not in
                    # the initialization blob dictionary, then load source blob
                    # 'foo' into destination blob '_[xyz]_foo'
                    src_name = unscoped_param_name[
                        unscoped_param_name.find(']_') + 2:]
                else:
                    src_name = unscoped_param_name
                if src_name not in src_blobs:
                    logger.info('{:s} not found'.format(src_name))
                    continue
                dst_name = core.ScopedName(unscoped_param_name)
                has_momentum = src_name + '_momentum' in src_blobs
                has_momentum_str = ' [+ momentum]' if has_momentum else ''
                logger.info('{:s}{:} loaded from weights file into {:s}: {}'.
                            format(
                                src_name, has_momentum_str,
                                dst_name, src_blobs[src_name].shape))
                pretrained_w = src_blobs[src_name]
                if dst_name in ws_blobs:
                    # If the blob is already in the workspace, make sure that it
                    # matches the shape of the loaded blob
                    ws_blob = workspace.FetchBlob(dst_name)
                    if ws_blob.shape != src_blobs[src_name].shape:
                        pretrained_w = inflate_weights(
                            pretrained_w, ws_blob, src_name, src_blobs)
                workspace.FeedBlob(
                    dst_name,
                    pretrained_w.astype(np.float32, copy=False))
                if has_momentum and not is_first_init:
                    # when feeding momentum, we're probably resuming from
                    # previous checkpoint. So all the inflated stuff won't be
                    # needed in that case
                    workspace.FeedBlob(
                        dst_name + '_momentum',
                        src_blobs[src_name + '_momentum'].astype(
                            np.float32, copy=False))

    # Add _rm/_riv BN mean/var params, in case the pre-trained model contains it.
    # Needed to test the scratch trained models.
    for src_name in src_blobs.keys():
        if src_name.endswith('_rm') or src_name.endswith('_riv'):
            with core.NameScope('gpu_0'):
                with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
                    dst_name = core.ScopedName(src_name)
                    workspace.FeedBlob(dst_name, src_blobs[src_name])
                    logger.info('Loaded BN param {}'.format(src_name))

    # We preserve blobs that are in the weights file but not used by the current
    # model. We load these into CPU memory under the '__preserve__/' namescope.
    # These blobs will be stored when saving a model to a weights file. This
    # feature allows for alternating optimization of Faster R-CNN in which blobs
    # unused by one step can still be preserved forward and used to initialize
    # another step.
    for src_name in src_blobs.keys():
        if (src_name not in unscoped_param_names and
                not src_name.endswith('_momentum') and
                src_blobs[src_name] is not None):
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CPU)):
                workspace.FeedBlob(
                    '__preserve__/{:s}'.format(src_name), src_blobs[src_name])
                logger.info(
                    '{:s} preserved in workspace (unused)'.format(src_name))


def save_model_to_weights_file(weights_file, model):
    """Stash model weights in a dictionary and pickle them to a file. We map
    GPU device scoped names to unscoped names (e.g., 'gpu_0/conv1_w' ->
    'conv1_w').
    """
    logger.info(
        'Saving parameters and momentum to {}'.format(
            os.path.abspath(weights_file)))
    blobs = {}
    # Save all parameters
    for param in model.params:
        scoped_name = str(param)
        unscoped_name = utils.blob.unscope_name(scoped_name)
        if unscoped_name not in blobs:
            logger.debug(' {:s} -> {:s}'.format(scoped_name, unscoped_name))
            blobs[unscoped_name] = workspace.FetchBlob(scoped_name)
    # Save momentum
    for param in model.TrainableParams():
        scoped_name = str(param) + '_momentum'
        unscoped_name = utils.blob.unscope_name(scoped_name)
        if unscoped_name not in blobs:
            logger.debug(' {:s} -> {:s}'.format(scoped_name, unscoped_name))
            blobs[unscoped_name] = workspace.FetchBlob(scoped_name)
    # Save preserved blobs
    for scoped_name in workspace.Blobs():
        if scoped_name.startswith('__preserve__/'):
            unscoped_name = utils.blob.unscope_name(scoped_name)
            if unscoped_name not in blobs:
                logger.debug(
                    ' {:s} -> {:s} (preserved)'.format(
                        scoped_name, unscoped_name))
                blobs[unscoped_name] = workspace.FetchBlob(scoped_name)
    # Save the _rm/_riv for the models with batch norm
    for scoped_name in workspace.Blobs():
        if scoped_name.endswith('_rm') or scoped_name.endswith('_riv'):
            unscoped_name = utils.blob.unscope_name(scoped_name)
            if unscoped_name not in blobs:
                logger.debug(
                    ' {:s} -> {:s} (preserved)'.format(
                        scoped_name, unscoped_name))
                blobs[unscoped_name] = workspace.FetchBlob(scoped_name)
    cfg_yaml = yaml.dump(cfg)
    robust_pickle_dump(dict(blobs=blobs, cfg=cfg_yaml), weights_file)


def broadcast_parameters(model):
    """Copy parameter blobs from GPU 0 over the corresponding parameter blobs
    on GPUs 1 through cfg.NUM_GPUS - 1.
    """
    if cfg.NUM_GPUS == 1:
        # no-op if only running on a single GPU
        return

    # TODO(rbg): replace with NCCLBroadcast when it's working
    # This doesn't work right now:
    # with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, 0)):
    #     workspace.RunOperatorOnce(
    #         core.CreateOperator(
    #             'NCCLBroadcast', model.params, model.params, root=0))
    def _do_broadcast(all_blobs):
        assert len(all_blobs) % cfg.NUM_GPUS == 0, 'This should not happen.'
        blobs_per_gpu = int(len(all_blobs) / cfg.NUM_GPUS)
        for i in range(blobs_per_gpu):
            blobs = [p for p in all_blobs[i::blobs_per_gpu]]
            data = workspace.FetchBlob(blobs[0])
            logger.debug('Broadcasting {} to'.format(str(blobs[0])))
            for i, p in enumerate(blobs[1:]):
                logger.debug(' |-> {}'.format(str(p)))
                with core.DeviceScope(
                        core.DeviceOption(caffe2_pb2.CUDA, i + 1)):
                    workspace.FeedBlob(p, data)

    _do_broadcast(model.params)
    _do_broadcast([b + '_momentum' for b in model.TrainableParams()])


def sum_multi_gpu_blob(blob_name):
    """Return the sum of a scalar blob held on multiple GPUs."""
    val = 0
    for i in range(cfg.NUM_GPUS):
        val += float(workspace.FetchBlob('gpu_{}/{}'.format(i, blob_name)))
    return val


def average_multi_gpu_blob(blob_name):
    """Return the average of a scalar blob held on multiple GPUs."""
    return sum_multi_gpu_blob(blob_name) / cfg.NUM_GPUS


def print_net(model, namescope='gpu_0'):
    logger.info('Printing model: {}'.format(model.net.Name()))
    op_list = model.net.Proto().op
    for op in op_list:
        input_name = op.input
        # For simplicity: only print the first output
        # Not recommended if there are split layers
        output_name = str(op.output[0])
        op_type = op.type
        op_name = op.name

        if namescope is None or output_name.startswith(namescope):
            # Only print the forward pass network
            if output_name.find('grad') >= 0:
                break

            output_shape = workspace.FetchBlob(output_name).shape
            first_blob = True
            op_label = op_type + (op_name if op_name == '' else ':' + op_name)
            suffix = ' ------- (op: {})'.format(op_label)
            for j in range(len(input_name)):
                if input_name[j] in model.params:
                    continue
                input_blob = workspace.FetchBlob(input_name[j])
                if isinstance(input_blob, np.ndarray):
                    input_shape = input_blob.shape
                    logger.info('{:28s}: {:20s} => {:28s}: {:20s}{}'.format(
                        utils.blob.unscope_name(str(input_name[j])),
                        '{}'.format(input_shape),
                        utils.blob.unscope_name(str(output_name)),
                        '{}'.format(output_shape),
                        suffix))
                    if first_blob:
                        first_blob = False
                        suffix = ' ------|'
    logger.info('End of model: {}'.format(model.net.Name()))


def configure_bbox_reg_weights(model, saved_cfg):
    if 'MODEL' not in saved_cfg or 'BBOX_REG_WEIGHTS' not in saved_cfg.MODEL:
        logger.warning('Model from weights file was trained before config key '
                       'MODEL.BBOX_REG_WEIGHTS was added. Forcing '
                       'MODEL.BBOX_REG_WEIGHTS = (1., 1., 1., 1.) to ensure '
                       'correct **inference** behavior.')
        cfg.MODEL.BBOX_REG_WEIGHTS = (1., 1., 1., 1.)
        logger.info('New config:')
        logger.info(pprint.pformat(cfg))
        assert not model.train, 'This mode should only be used for inference'
