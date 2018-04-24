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

import copy
from caffe2.python import core, workspace, muji

from core.config import cfg
from modeling.detector import DetectionModelHelper
from roi_data.loader import RoIDataLoader
import roi_data.minibatch
from modeling.generate_anchors import generate_anchors
import utils.blob as blob_utils

# Implemented model types
import modeling.VGG16 as VGG16
import modeling.VGG_CNN_M_1024 as VGG_CNN_M_1024
import modeling.ResNet as ResNet
import modeling.FPN as FPN
import modeling.FPN3D as FPN3D  # NoQA (used in yaml configs)
import modeling.ResNet3D as ResNet3D  # NoQA (used in yaml configs)
import modeling.head_builder as head_builder  # NoQA (used in yaml configs)
import modeling.mask_rcnn_heads as mask_rcnn_heads  # NoQA
import modeling.keypoint_rcnn_heads as keypoint_rcnn_heads  # NoQA

import logging
logger = logging.getLogger(__name__)


def get_func(func_name):
    """Helper to return a function object by name."""
    try:
        parts = func_name.split('.')
        res = globals()[parts[0]]
        for part in parts[1:]:
            res = getattr(res, part)
        return res
    except Exception:
        logger.error('Failed to find function: {}'.format(func_name))
        raise


def create(model_name, train=False, init_params=None):
    """Generic model creation function that dispatches to specific model
    building functions.
    Args:
        train (bool): Set true if training
        init_params (bool or None): Set to true if force initialize the network
            with random weights (even at test time). Normally will init only
            at train time.
    """
    return get_func(model_name)(init_model(model_name, train, init_params))


# ---------------------------------------------------------------------------- #
# Generic recomposable model builders
#
# For example, you can create a Fast R-CNN model with VGG_CNN_M_1024 by with the
# configuration:
#
# MODEL:
#   TYPE: fast_rcnn
#   CONV_BODY: VGG_CNN_M_1024.add_VGG_CNN_M_1024_conv5_body
#   ROI_HEAD: VGG_CNN_M_1024.add_VGG_CNN_M_1024_roi_fc_head
#
# ---------------------------------------------------------------------------- #

def fast_rcnn(model):
    return build_generic_fast_rcnn_model(
        model,
        get_func(cfg.MODEL.CONV_BODY),
        get_func(cfg.MODEL.ROI_HEAD))


def fast_rcnn_frozen_features(model):
    return build_generic_fast_rcnn_model(
        model,
        get_func(cfg.MODEL.CONV_BODY),
        get_func(cfg.MODEL.ROI_HEAD),
        freeze_conv_body=True)


def rpn(model):
    return build_generic_rpn_model(
        model,
        get_func(cfg.MODEL.CONV_BODY))


def rpn_frozen_features(model):
    return build_generic_rpn_model(
        model,
        get_func(cfg.MODEL.CONV_BODY),
        freeze_conv_body=True)


def fpn_rpn(model):
    return build_generic_rpn_model(
        model,
        get_func(cfg.MODEL.CONV_BODY))


def fpn_rpn_frozen_features(model):
    return build_generic_rpn_model(
        model,
        get_func(cfg.MODEL.CONV_BODY),
        freeze_conv_body=True)


def faster_rcnn(model):
    assert cfg.MODEL.FASTER_RCNN
    return build_generic_fast_rcnn_model(
        model,
        get_func(cfg.MODEL.CONV_BODY),
        get_func(cfg.MODEL.ROI_HEAD))


def rfcn(model):
    return build_generic_rfcn_model(
        model,
        get_func(cfg.MODEL.CONV_BODY))


def mask_rcnn(model):
    return build_generic_fast_rcnn_model(
        model,
        get_func(cfg.MODEL.CONV_BODY),
        get_func(cfg.MODEL.ROI_HEAD),
        add_roi_mask_head_func=get_func(cfg.MRCNN.MASK_HEAD_NAME))


def mask_rcnn_frozen_features(model):
    return build_generic_fast_rcnn_model(
        model,
        get_func(cfg.MODEL.CONV_BODY),
        get_func(cfg.MODEL.ROI_HEAD),
        add_roi_mask_head_func=get_func(cfg.MRCNN.MASK_HEAD_NAME),
        freeze_conv_body=True)


def keypoint_rcnn(model):
    return build_generic_fast_rcnn_model(
        model,
        get_func(cfg.MODEL.CONV_BODY),
        get_func(cfg.MODEL.ROI_HEAD),
        add_roi_keypoint_head_func=get_func(cfg.KRCNN.ROI_KEYPOINTS_HEAD))


def keypoint_rcnn_frozen_features(model):
    return build_generic_fast_rcnn_model(
        model,
        get_func(cfg.MODEL.CONV_BODY),
        get_func(cfg.MODEL.ROI_HEAD),
        add_roi_keypoint_head_func=get_func(cfg.KRCNN.ROI_KEYPOINTS_HEAD),
        freeze_conv_body=True)


def mask_and_keypoint_rcnn(model):
    return build_generic_fast_rcnn_model(
        model,
        get_func(cfg.MODEL.CONV_BODY),
        get_func(cfg.MODEL.ROI_HEAD),
        add_roi_mask_head_func=get_func(cfg.MRCNN.MASK_HEAD_NAME),
        add_roi_keypoint_head_func=get_func(cfg.KRCNN.ROI_KEYPOINTS_HEAD))


# ---------------------------------------------------------------------------- #
# Helper functions for building various re-usable network bits
# ---------------------------------------------------------------------------- #

def build_generic_fast_rcnn_model(
        model, add_conv_body_func, add_roi_frcn_head_func,
        add_roi_mask_head_func=None, add_roi_keypoint_head_func=None,
        freeze_conv_body=False):
    def _single_gpu_build_func(model):
        """Builds the model on a single GPU. Can be called in a loop over GPUs
        with name and device scoping to create a data parallel model."""
        # For training we define one net that contains all ops
        # For inference, we split the graph into two nets: a standard fast r-cnn
        # net and a mask prediction net; the mask net is only applied to a
        # subset of high-scoring detections
        is_inference = not model.train

        # Some generic tensors
        model.ConstantFill([], 'zero', shape=[1], value=0)
        model.ConstantFill([], 'minus1', shape=[1], value=-1)

        # Add the conv body
        blob_conv, dim_conv, spatial_scale_conv = add_conv_body_func(model)
        if freeze_conv_body:
            for b in blob_ref_to_list(blob_conv):
                model.StopGradient(b, b)

        # Convert from 3D blob to 2D, in case of videos to attach a 2D head
        # (not necessarily will happen though)
        if cfg.MODEL.VIDEO_ON:
            blob_conv = time_pool_blobs(
                blob_conv, model, cfg.VIDEO.BODY_HEAD_LINK)

        if is_inference:
            # Create a net that can be used to compute the conv body only on an
            # image (no RPN or heads / branches)
            model.conv_body_net = model.net.Clone('conv_body_net')

        # Select the FPN lib, based on whether the head is 3D or 2D
        if cfg.MODEL.VIDEO_ON and cfg.VIDEO.BODY_HEAD_LINK == '':
            FPN_lib = FPN3D
            head_3d = True
            out_time_dim = cfg.VIDEO.NUM_FRAMES_MID
        else:
            FPN_lib = FPN
            head_3d = False
            out_time_dim = 1

        # Add the RPN branch
        if cfg.MODEL.FASTER_RCNN:
            if cfg.FPN.FPN_ON:
                FPN_lib.add_fpn_rpn_outputs(
                    model, blob_conv, dim_conv, spatial_scale_conv,
                    time_dim=out_time_dim)
                model.CollectAndDistributeFpnRpnProposals()
            else:
                add_rpn_outputs(model, blob_conv, dim_conv, spatial_scale_conv,
                                nd=head_3d, time_dim=out_time_dim)

        if cfg.FPN.FPN_ON:
            # Code only supports case when RPN and ROI min levels are the same
            assert cfg.FPN.RPN_MIN_LEVEL == cfg.FPN.ROI_MIN_LEVEL
            # FPN RPN max level might be > FPN ROI max level in which case we
            # need to discard some leading conv blobs (blobs are ordered from
            # max level to min level)
            num_roi_levels = cfg.FPN.ROI_MAX_LEVEL - cfg.FPN.ROI_MIN_LEVEL + 1
            blob_conv = blob_conv[-num_roi_levels:]
            spatial_scale_conv = spatial_scale_conv[-num_roi_levels:]

        # Add the Fast R-CNN branch
        blob_frcn, dim_frcn, spatial_scale_frcn = add_roi_frcn_head_func(
            model, blob_conv, dim_conv, spatial_scale_conv)
        add_fast_rcnn_outputs(model, blob_frcn, dim_frcn, is_head_3d=head_3d)

        # Add the mask branch
        if cfg.MODEL.MASK_ON:
            if is_inference:
                bbox_net = copy.deepcopy(model.net.Proto())

            # Add the mask branch
            blob_mrcn, dim_mrcn, _ = add_roi_mask_head_func(
                model, blob_conv, dim_conv, spatial_scale_conv)
            blob_mask = add_mask_rcnn_outputs(model, blob_mrcn, dim_mrcn)

            if is_inference:
                # Extract the mask prediction net, store it as its own network,
                # then restore the primary net to the bbox-only network
                model.mask_net, blob_mask = get_suffix_net(
                    'mask_net', bbox_net.op, model.net, [blob_mask])
                model.net._net = bbox_net

        # Add the keypoint branch
        if cfg.MODEL.KEYPOINTS_ON:
            if is_inference:
                bbox_net = copy.deepcopy(model.net.Proto())

            blob_krcnn, dim_krcnn, _ = add_roi_keypoint_head_func(
                model, blob_conv, dim_conv, spatial_scale_conv)
            blob_keypoint = add_heatmap_outputs(
                model, blob_krcnn, dim_krcnn,
                time_dim=out_time_dim, is_head_3d=head_3d)

            if is_inference:
                model.keypoint_net, keypoint_blob_out = get_suffix_net(
                    'keypoint_net', bbox_net.op, model.net, [blob_keypoint])
                model.net._net = bbox_net

        if model.train:
            loss_gradients = add_fast_rcnn_losses(model, time_dim=out_time_dim)
            if cfg.MODEL.MASK_ON:
                loss_gradients.update(add_mask_rcnn_losses(model, blob_mask,
                                                           time_dim=out_time_dim))
            if cfg.MODEL.KEYPOINTS_ON:
                loss_gradients.update(add_heatmap_losses(model, time_dim=out_time_dim))
            if cfg.MODEL.FASTER_RCNN:
                if cfg.FPN.FPN_ON:
                    # The loss function is shared between 2D and 3D FPN
                    loss_gradients.update(FPN.add_fpn_rpn_losses(
                        model, time_dim=out_time_dim))
                    if cfg.VIDEO.PREDICT_RPN_BOX_VIS:
                        loss_gradients.update(FPN.add_fpn_rpn_vis_losses(
                            model, time_dim=out_time_dim))
                else:
                    loss_gradients.update(add_rpn_losses(
                        model, time_dim=out_time_dim))
                    if cfg.VIDEO.PREDICT_RPN_BOX_VIS:
                        loss_gradients.update(add_rpn_vis_losses(
                            model, time_dim=out_time_dim))
        return loss_gradients if model.train else None

    build_data_parallel_model(model, _single_gpu_build_func)
    return model


def build_generic_rpn_model(model, add_conv_body_func, freeze_conv_body=False):
    def _single_gpu_build_func(model):
        """Builds the model on a single GPU. Can be called in a loop over GPUs
        with name and device scoping to create a data parallel model."""
        # Some generic tensors
        model.ConstantFill([], 'zero', shape=[1], value=0)
        model.ConstantFill([], 'minus1', shape=[1], value=-1)

        blob, dim, spatial_scale = add_conv_body_func(model)
        if freeze_conv_body:
            model.StopGradient(blob, blob)
        if cfg.MODEL.VIDEO_ON:
            blob = time_pool_blobs(
                blob, model, cfg.VIDEO.BODY_HEAD_LINK)
        if not model.train:
            # Create a net that can be used to compute the conv body only on an
            # image (no RPN or heads / branches)
            model.conv_body_net = model.net.Clone('conv_body_net')
        if cfg.MODEL.VIDEO_ON and cfg.VIDEO.BODY_HEAD_LINK == '':
            FPN_lib = FPN3D
            head_3d = True
            out_time_dim = cfg.VIDEO.NUM_FRAMES_MID
        else:
            FPN_lib = FPN
            head_3d = False
            out_time_dim = 1
        if cfg.FPN.FPN_ON:
            FPN_lib.add_fpn_rpn_outputs(model, blob, dim, spatial_scale,
                                        time_dim=out_time_dim)
            if model.train:
                loss_gradients = FPN.add_fpn_rpn_losses(model)
        else:
            add_rpn_outputs(model, blob, dim, spatial_scale, nd=head_3d,
                            time_dim=out_time_dim)
            if model.train:
                loss_gradients = add_rpn_losses(model,
                                                time_dim=out_time_dim)
        return loss_gradients if model.train else None

    build_data_parallel_model(model, _single_gpu_build_func)
    return model


def build_generic_rfcn_model(model, add_conv_body_func, dim_reduce=None):
    def _single_gpu_build_func(model):
        """Builds the model on a single GPU. Can be called in a loop over GPUs
        with name and device scoping to create a data parallel model."""
        blob, dim, spatial_scale = add_conv_body_func(model)
        if cfg.MODEL.VIDEO_ON:
            raise NotImplementedError('Not looked at this style models')
        if not model.train:
            # Create a net that can be used to compute the conv body only on an
            # image (no RPN or heads / branches)
            model.conv_body_net = model.net.Clone('conv_body_net')
        add_rfcn_outputs(model, blob, dim, dim_reduce, spatial_scale)
        if model.train:
            loss_gradients = add_fast_rcnn_losses(model)
        return loss_gradients if model.train else None

    build_data_parallel_model(model, _single_gpu_build_func)
    return model


# ---------------------------------------------------------------------------- #
# Network inputs
# ---------------------------------------------------------------------------- #

def add_inputs(model, roidb=None):
    """Add network input ops. To be called *after* model_bulder.create()."""
    # Implementation notes:
    #   Typically, one would create the input ops and then the rest of the net.
    #   However, creating the input ops depends on loading the dataset, which
    #   can take a few minutes for COCO.
    #   We prefer to avoid waiting so debugging can fail fast.
    #   Thus, we create the net *without input ops* prior to loading the
    #   dataset, and then add the input ops after loading the dataset.
    #   Since we defer input op creation, we need to do a little bit of surgery
    #   to place the input ops at the start of the network op list.
    if roidb is not None:
        # Make debugging easier when NUM_GPUS is 1 by only using one worker
        # thread for loading mini-batches
        num_workers = 1 if cfg.NUM_GPUS == 1 else cfg.NUM_WORKERS
        model.roi_data_loader = RoIDataLoader(
            roidb, num_workers=num_workers, num_enqueuers=1,
            minibatch_queue_size=cfg.TRAIN.MINIBATCH_QUEUE_SIZE)
    orig_num_op = len(model.net._net.op)
    for gpu_id in range(cfg.NUM_GPUS):
        with core.NameScope('gpu_{}'.format(gpu_id)):
            with core.DeviceScope(muji.OnGPU(gpu_id)):
                if model.train:
                    add_train_inputs(model)
                else:
                    add_test_inputs(model)
    # A little op surgery to move input ops to the start of the net
    diff = len(model.net._net.op) - orig_num_op
    new_op = model.net._net.op[-diff:] + model.net._net.op[:-diff]
    del model.net._net.op[:]
    model.net._net.op.extend(new_op)


def add_train_inputs(model):
    blob_names = model.roi_data_loader.get_output_names()
    for blob_name in blob_names:
        workspace.CreateBlob(core.ScopedName(blob_name))
    model.net.DequeueBlobs(model.roi_data_loader._blobs_queue_name, blob_names)


def add_test_inputs(model):
    blob_names = roi_data.minibatch.get_minibatch_blob_names()
    for blob_name in blob_names:
        workspace.CreateBlob(core.ScopedName(blob_name))


# ---------------------------------------------------------------------------- #
# Fast R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

def add_fast_rcnn_outputs(model, blob_in, dim, is_head_3d):
    if is_head_3d:
        # As per the changes to ResNet head, the output will be a 3D blob
        # so that I can run 3D convolutions on it. But be careful to output a 2D
        # blob from here
        cls_score = model.ConvNd(
            blob_in, 'cls_score_1', dim, model.num_classes,
            [1, 1, 1], pads=2 * [0, 0, 0], strides=[1, 1, 1],
            weight_init=('GaussianFill', {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.}))
        # Does not support inplace operations! WOW
        model.ReduceBackMean(model.ReduceBackMean(model.ReduceBackMean(
            cls_score, 'cls_score_2'), 'cls_score_3'), 'cls_score')
    else:
        # Original code
        model.FC(
            blob_in, 'cls_score', dim, model.num_classes,
            weight_init=('GaussianFill', {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.}))
    if not model.train:  # == if test
        # Only add softmax when testing; during training the softmax is combined
        # with the label cross entropy loss for numerical stability
        model.Softmax('cls_score', 'cls_prob', engine='CUDNN')
    if is_head_3d:
        model.ConvNd(
            blob_in, 'bbox_pred_1', dim,
            4 * model.num_classes, [1, 1, 1],
            pads=2 * [0, 0, 0], strides=[1, 1, 1],
            weight_init=('GaussianFill', {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.}))
        # Convert into the format bbox losses expect (Same as RPN)
        # Convert the Bx(4C)xTxHxW -> BxCx4xTxHxW
        model.ExpandDims('bbox_pred_1', 'bbox_pred_2', dims=[2])
        model.Reshape(['bbox_pred_2'], ['bbox_pred_3', model.net.NextName()],
                      shape=(0, -1, 4, 0, 0, 0))
        # Convert the BxCx4xTxHxW -> BxCxTx4xHxW
        model.Transpose('bbox_pred_3', 'bbox_pred_4',
                        axes=(0, 1, 3, 2, 4, 5))
        # Convert the BxCxTx4xHxW -> Bx(C*T*4)xHxW
        batch_size = model.GetShapeDimIdx(blob_in, 0)
        ht = model.GetShapeDimIdx(blob_in, 3)
        wd = model.GetShapeDimIdx(blob_in, 4)
        final_shape = model.GetNewShape(batch_size, -1, ht, wd)
        model.Reshape(['bbox_pred_4', final_shape],
                      ['bbox_pred_5', model.net.NextName()])
        # Does not support inplace operations! WOW
        model.ReduceBackMean(model.ReduceBackMean(
            'bbox_pred_5', 'bbox_pred_6'), 'bbox_pred')
    else:
        model.FC(
            blob_in, 'bbox_pred', dim, model.num_classes * 4,
            weight_init=('GaussianFill', {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.}))


def add_fast_rcnn_losses(model, time_dim=1):
    cls_prob, loss_cls = model.net.SoftmaxWithLoss(
        ['cls_score', 'labels_int32'], ['cls_prob', 'loss_cls'],
        scale=1. / cfg.NUM_GPUS)
    loss_bbox = model.net.SmoothL1Loss(
        ['bbox_pred', 'bbox_targets', 'bbox_inside_weights',
         'bbox_outside_weights'], 'loss_bbox',
        scale=1. / cfg.NUM_GPUS / time_dim)
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_cls, loss_bbox])
    model.Accuracy(['cls_prob', 'labels_int32'], 'accuracy_cls')
    model.losses = list(set(model.losses + ['loss_cls', 'loss_bbox']))
    model.metrics = list(set(model.metrics + ['accuracy_cls']))
    return loss_gradients


# ---------------------------------------------------------------------------- #
# RPN and Faster R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

def add_rpn_outputs(model, blob_in, dim_in, spatial_scale, nd=False, time_dim=1):
    anchors = generate_anchors(
        stride=1. / spatial_scale,
        sizes=cfg.RPN.SIZES,
        aspect_ratios=cfg.RPN.ASPECT_RATIOS,
        time_dim=time_dim)
    num_anchors = anchors.shape[0]
    dim_out = dim_in
    # RPN hidden representation
    if nd:
        model.ConvNd(
            blob_in, 'conv_rpn', dim_in, dim_out,
            [cfg.VIDEO.TIME_KERNEL_DIM.HEAD_RPN, 3, 3],
            pads=2 * [cfg.VIDEO.TIME_KERNEL_DIM.HEAD_RPN // 2, 1, 1], strides=[1, 1, 1],
            weight_init=('GaussianFill', {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.}))
    else:
        model.Conv(
            blob_in, 'conv_rpn', dim_in, dim_out, 3, pad=1, stride=1,
            weight_init=('GaussianFill', {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.}))
    model.Relu('conv_rpn', 'conv_rpn')
    # Proposal classification scores
    conv_rpn = 'conv_rpn'
    # if nd:
    #     # Convert to 2D.
    #     model.MoveTimeToChannelDim('conv_rpn', 'conv_rpn_timepooled')
    #     # model.TimePool('conv_rpn', 'conv_rpn_timepooled', pool_type='avg')
    #     conv_rpn = 'conv_rpn_timepooled'
    #     final_dim_in = dim_out * time_dim
    #     # final_dim_in = dim_out
    if nd:
        model.ConvNd(
            conv_rpn, 'rpn_cls_logits_1', dim_out, num_anchors,
            [1, 1, 1], pads=2 * [0, 0, 0], strides=[1, 1, 1],
            weight_init=('GaussianFill', {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.}))
        model.TimePool('rpn_cls_logits_1', 'rpn_cls_logits', pool_type='avg')
    else:
        model.Conv(
            conv_rpn, 'rpn_cls_logits', dim_out, num_anchors, 1,
            pad=0, stride=1,
            weight_init=('GaussianFill', {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.}))
    # Proposal bbox regression deltas
    if nd:
        model.ConvNd(
            conv_rpn, 'rpn_bbox_pred_1', dim_out,
            4 * num_anchors, [1, 1, 1],
            pads=2 * [0, 0, 0], strides=[1, 1, 1],
            weight_init=('GaussianFill', {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.}))
        # Convert into the format RPN losses expect
        model.ExpandDims('rpn_bbox_pred_1', 'rpn_bbox_pred_2', dims=[2])
        model.Reshape(['rpn_bbox_pred_2'], ['rpn_bbox_pred_3', model.net.NextName()],
                      shape=(0, -1, 4, 0, 0, 0))
        model.Transpose('rpn_bbox_pred_3', 'rpn_bbox_pred_4',
                        axes=(0, 1, 3, 2, 4, 5))
        batch_size = model.GetShapeDimIdx(conv_rpn, 0)
        ht = model.GetShapeDimIdx(conv_rpn, 3)
        wd = model.GetShapeDimIdx(conv_rpn, 4)
        final_shape = model.GetNewShape(batch_size, -1, ht, wd)
        model.Reshape(['rpn_bbox_pred_4', final_shape],
                      ['rpn_bbox_pred', model.net.NextName()])
    else:
        model.Conv(
            conv_rpn, 'rpn_bbox_pred', dim_out,
            4 * num_anchors, 1, pad=0, stride=1,
            weight_init=('GaussianFill', {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.}))
    # Proposal visibility classification scores
    # TODO: need to use in future
    # model.Conv(
    #     conv_rpn, 'rpn_vis_logits', final_dim_in,
    #     num_anchors * time_dim, 1, pad=0, stride=1,
    #     weight_init=('GaussianFill', {'std': 0.01}),
    #     bias_init=('ConstantFill', {'value': 0.}))

    if cfg.MODEL.FASTER_RCNN or (cfg.MODEL.RPN_ONLY and not model.train):
        # Add op that generates RPN proposals
        # The proposals are needed when generating proposals from an RPN-only
        # model (but *not* when training it) or when training or testing a
        # Faster R-CNN model
        model.net.Sigmoid('rpn_cls_logits', 'rpn_cls_probs')
        model.GenerateProposals(
            ['rpn_cls_probs', 'rpn_bbox_pred', 'im_info'],
            ['rpn_rois', 'rpn_roi_probs'],
            anchors=anchors,
            spatial_scale=spatial_scale)

        if cfg.VIDEO.DEBUG_USE_RPN_GT:
            model.net.SpatialNarrowAs(
                ['rpn_labels_int32_wide', 'rpn_cls_logits'], 'rpn_labels_int32')
            for key in ('targets', 'inside_weights', 'outside_weights'):
                model.net.SpatialNarrowAs(
                    ['rpn_bbox_' + key + '_wide', 'rpn_bbox_pred'],
                    'rpn_bbox_' + key)
            model.GenerateProposals(
                ['rpn_labels_int32', 'rpn_bbox_targets', 'im_info'],
                ['rpn_rois', 'rpn_roi_probs'],
                anchors=anchors,
                spatial_scale=spatial_scale)

    if cfg.MODEL.FASTER_RCNN:
        if model.train:
            # Add op that generates training labels for in-network RPN proposals
            model.GenerateProposalLabels(['rpn_rois', 'roidb', 'im_info'])
        else:
            # Alias rois to rpn_rois if not training
            model.net.Alias('rpn_rois', 'rois')


def add_rpn_losses(model, time_dim=1):
    # Spatially narrow the full-sized RPN label arrays to match the feature map
    # shape
    if not model.BlobExists('rpn_labels_int32'):
        model.net.SpatialNarrowAs(
            ['rpn_labels_int32_wide', 'rpn_cls_logits'], 'rpn_labels_int32')
    for key in ('targets', 'inside_weights', 'outside_weights'):
        if not model.BlobExists('rpn_bbox_' + key):
            model.net.SpatialNarrowAs(
                ['rpn_bbox_' + key + '_wide', 'rpn_bbox_pred'],
                'rpn_bbox_' + key)
    loss_rpn_cls = model.net.SigmoidCrossEntropyLoss(
        ['rpn_cls_logits', 'rpn_labels_int32'], 'loss_rpn_cls',
        scale=1. / cfg.NUM_GPUS)
    loss_rpn_bbox = model.net.SmoothL1Loss(
        ['rpn_bbox_pred', 'rpn_bbox_targets', 'rpn_bbox_inside_weights',
         'rpn_bbox_outside_weights'],
        'loss_rpn_bbox',
        beta=1. / 9.,
        scale=1. / cfg.NUM_GPUS / time_dim)
    loss_gradients = blob_utils.get_loss_gradients(
        model, [loss_rpn_cls, loss_rpn_bbox])
    model.losses = list(set(model.losses + ['loss_rpn_cls', 'loss_rpn_bbox']))
    return loss_gradients


def add_rpn_vis_losses(model, time_dim=1):
    # Spatially narrow the full-sized RPN label arrays to match the feature map
    # shape
    model.net.SpatialNarrowAs(
        ['rpn_vis_labels_int32_wide', 'rpn_vis_cls_logits'], 'rpn_vis_labels_int32')
    loss_rpn_vis_cls = model.net.SigmoidCrossEntropyLoss(
        ['rpn_vis_cls_logits', 'rpn_vis_labels_int32'], 'loss_rpn_vis_cls',
        scale=1. / cfg.NUM_GPUS / time_dim)
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_rpn_vis_cls])
    model.losses = list(set(model.losses + ['loss_rpn_vis_cls']))
    return loss_gradients


# ---------------------------------------------------------------------------- #
# R-FCN outputs and losses
# ---------------------------------------------------------------------------- #

def add_rfcn_outputs(model, blob_in, dim_in, dim_reduce, spatial_scale):
    if dim_reduce is not None:
        # Optional dim reduction
        blob_in = model.Conv(
            blob_in, 'conv_dim_reduce', dim_in, dim_reduce, 1,
            pad=0, stride=1,
            weight_init=('GaussianFill', {'std': 0.01}),
            bias_init=('ConstantFill', {'value': 0.}))
        blob_in = model.Relu(blob_in, blob_in)
        dim_in = dim_reduce
    # Classification conv
    model.Conv(
        blob_in, 'conv_cls', dim_in,
        model.num_classes * cfg.MODEL.PS_GRID_SIZE ** 2, 1, pad=0, stride=1,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.}))
    # # Bounding-box regression conv
    num_bbox_reg_classes = (
        2 if cfg.MODEL.CLS_AGNOSTIC_BBOX_REG else model.num_classes)
    model.Conv(
        blob_in, 'conv_bbox_pred', dim_in,
        4 * num_bbox_reg_classes * cfg.MODEL.PS_GRID_SIZE ** 2, 1,
        pad=0, stride=1,
        weight_init=('GaussianFill', {'std': 0.01}),
        bias_init=('ConstantFill', {'value': 0.}))
    # Classification PS RoI pooling
    model.net.PSRoIPool(
        ['conv_cls', 'rois'], ['psroipooled_cls', '_mapping_channel_cls'],
        group_size=cfg.MODEL.PS_GRID_SIZE, output_dim=model.num_classes,
        spatial_scale=spatial_scale)
    model.AveragePool(
        'psroipooled_cls', 'cls_score', kernel=cfg.MODEL.PS_GRID_SIZE)
    if not model.train:
        model.Softmax('cls_score', 'cls_prob', engine='CUDNN')
    # Bbox regression PS RoI pooling
    model.net.PSRoIPool(
        ['conv_bbox_pred', 'rois'],
        ['psroipooled_bbox', '_mapping_channel_bbox'],
        group_size=cfg.MODEL.PS_GRID_SIZE, output_dim=4 * num_bbox_reg_classes,
        spatial_scale=spatial_scale)
    model.AveragePool(
        'psroipooled_bbox', 'bbox_pred', kernel=cfg.MODEL.PS_GRID_SIZE)


# ---------------------------------------------------------------------------- #
# Mask R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

def add_mask_rcnn_outputs(model, blob_in, dim):
    num_cls = cfg.MODEL.NUM_CLASSES if cfg.MRCNN.CLS_SPECIFIC_MASK else 1

    if cfg.MRCNN.USE_FC_OUTPUT:
        blob_out = model.FC(
            blob_in, 'mask_fcn_logits', dim,
            num_cls * cfg.MRCNN.RESOLUTION**2,
            weight_init=('GaussianFill', {'std': 0.001}),
            bias_init=('ConstantFill', {'value': 0.}))
    else:
        if cfg.MRCNN.UPSAMPLE_RATIO == 1:
            blob_out = model.Conv(
                blob_in, 'mask_fcn_logits', dim, num_cls, 1,
                pad=0, stride=1,
                weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
                bias_init=('ConstantFill', {'value': 0.}))
        else:  # upsample
            model.Conv(
                blob_in, 'mask_fcn_logits', dim, num_cls, 1,
                pad=0, stride=1,
                weight_init=(cfg.MRCNN.CONV_INIT, {'std': 0.001}),
                bias_init=('ConstantFill', {'value': 0.}))
            blob_out = model.BilinearInterpolation(
                'mask_fcn_logits', 'mask_fcn_logits_up',
                num_cls, num_cls,
                cfg.MRCNN.UPSAMPLE_RATIO)

    if not model.train:  # == if test
        blob_out = model.net.Sigmoid(blob_out, 'mask_fcn_probs')

    return blob_out


def add_mask_rcnn_losses(model, blob_mask):
    if cfg.MRCNN.CLS_SPECIFIC_MASK:
        loss_mask = model.net.SigmoidCrossEntropyLoss(
            [blob_mask, 'masks_int32'], 'loss_mask',
            scale=1. / cfg.NUM_GPUS * cfg.MRCNN.WEIGHT_LOSS_MASK)
    else:  # cls-agnostic; using sigmoid
        # logistic regression (binary)
        loss_mask = model.net.SigmoidCrossEntropyLoss(
            [blob_mask, 'masks_int32'], 'loss_mask',
            scale=1. / cfg.NUM_GPUS * cfg.MRCNN.WEIGHT_LOSS_MASK)
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_mask])
    model.losses = list(set(model.losses + ['loss_mask']))
    return loss_gradients


# ---------------------------------------------------------------------------- #
# Keypoint R-CNN outputs and losses
# ---------------------------------------------------------------------------- #

def add_heatmap_outputs(model, blob_in, dim, time_dim, is_head_3d):
    use_3d_deconv = is_head_3d and cfg.KRCNN.USE_3D_DECONV
    move_back_batch_to_time = False
    if is_head_3d and not cfg.KRCNN.USE_3D_DECONV:
        # Can't do upsamples etc in time, so convert to 2D blob
        if cfg.KRCNN.NO_3D_DECONV_TIME_TO_CH:
            original_time_dim = model.GetTemporalDim(blob_in)
            blob_in = model.MoveTimeToBatchDim(blob_in, None)
            time_dim = 1
            move_back_batch_to_time = True
        else:
            blob_in = model.MoveTimeToChannelDim(blob_in, None)
            dim = dim * time_dim

    # NxKxHxW
    if cfg.KRCNN.USE_DECONV:
        if use_3d_deconv:
            psize_spatial = int(cfg.KRCNN.DECONV_KERNEL / 2 - 1)
            psize_temporal = int(cfg.VIDEO.TIME_KERNEL_DIM.HEAD_KPS / 2 - 1)
            blob_in = model.ConvTranspose3D(
                blob_in, 'kps_deconv', dim, cfg.KRCNN.DECONV_DIM,
                [cfg.VIDEO.TIME_KERNEL_DIM.HEAD_KPS,
                 cfg.KRCNN.DECONV_KERNEL, cfg.KRCNN.DECONV_KERNEL],
                pads=2 * [psize_temporal, psize_spatial, psize_spatial],
                stride=[1, 2, 2],
                weight_init=('GaussianFill', {'std': 0.01}),
                bias_init=('ConstantFill', {'value': 0.}))
        else:
            blob_in = model.ConvTranspose(
                blob_in, 'kps_deconv', dim, cfg.KRCNN.DECONV_DIM,
                cfg.KRCNN.DECONV_KERNEL,
                pad=int(cfg.KRCNN.DECONV_KERNEL / 2 - 1),
                stride=2,
                group=time_dim,
                weight_init=('GaussianFill', {'std': 0.01}),
                bias_init=('ConstantFill', {'value': 0.}))
        model.Relu('kps_deconv', 'kps_deconv')
        dim = cfg.KRCNN.DECONV_DIM

    if cfg.KRCNN.UP_SCALE == 1:
        raise NotImplementedError(
            'Handle the 3D case. Skipping for now as I am lazy and dont want '
            'to implement something I might not even use. ')
        if not cfg.KRCNN.USE_DECONV_OUTPUT:
            blob_out = model.Conv(
                blob_in, 'kps_score', dim,
                cfg.KRCNN.NUM_KEYPOINTS * time_dim, 1,
                pad=0, stride=1,
                group=time_dim,
                weight_init=(cfg.KRCNN.CONV_INIT, {'std': 0.001}),
                bias_init=('ConstantFill', {'value': 0.}))
        else:
            blob_out = model.ConvTranspose(
                blob_in, 'kps_score', dim,
                cfg.KRCNN.NUM_KEYPOINTS * time_dim,
                cfg.KRCNN.DECONV_KERNEL,
                pad=int(cfg.KRCNN.DECONV_KERNEL / 2 - 1),
                stride=2,
                group=time_dim,
                weight_init=(cfg.KRCNN.CONV_INIT, {'std': 0.001}),
                bias_init=('ConstantFill', {'value': 0.}))
    else:  # upsample
        if not cfg.KRCNN.USE_DECONV_OUTPUT:
            if use_3d_deconv:
                model.ConvNd(
                    blob_in, 'kps_score_lowres', dim,
                    cfg.KRCNN.NUM_KEYPOINTS, [1, 1, 1],
                    pads=2 * [0, 0, 0], strides=[1, 1, 1],
                    weight_init=(cfg.KRCNN.CONV_INIT, {'std': 0.001}),
                    bias_init=('ConstantFill', {'value': 0.}))
            else:
                model.Conv(
                    blob_in, 'kps_score_lowres', dim,
                    cfg.KRCNN.NUM_KEYPOINTS * time_dim, 1,
                    pad=0, stride=1,
                    group=time_dim,
                    weight_init=(cfg.KRCNN.CONV_INIT, {'std': 0.001}),
                    bias_init=('ConstantFill', {'value': 0.}))
        else:
            if use_3d_deconv:
                psize_spatial = int(cfg.KRCNN.DECONV_KERNEL / 2 - 1)
                psize_temporal = int(cfg.VIDEO.TIME_KERNEL_DIM.HEAD_KPS / 2 - 1)
                model.ConvTranspose3D(
                    blob_in, 'kps_score_lowres_1', dim,
                    cfg.KRCNN.NUM_KEYPOINTS,
                    [cfg.VIDEO.TIME_KERNEL_DIM.HEAD_KPS, cfg.KRCNN.DECONV_KERNEL,
                     cfg.KRCNN.DECONV_KERNEL],
                    pads=2 * [psize_temporal, psize_spatial, psize_spatial],
                    strides=[1, 2, 2],
                    weight_init=(cfg.KRCNN.CONV_INIT, {'std': 0.001}),
                    bias_init=('ConstantFill', {'value': 0.}))
                model.MoveTimeToChannelDim('kps_score_lowres_1', 'kps_score_lowres')
            else:
                model.ConvTranspose(
                    blob_in, 'kps_score_lowres', dim,
                    cfg.KRCNN.NUM_KEYPOINTS * time_dim,
                    cfg.KRCNN.DECONV_KERNEL,
                    pad=int(cfg.KRCNN.DECONV_KERNEL / 2 - 1),
                    stride=2,
                    group=time_dim,
                    weight_init=(cfg.KRCNN.CONV_INIT, {'std': 0.001}),
                    bias_init=('ConstantFill', {'value': 0.}))

        blob_out = model.BilinearInterpolation(
            'kps_score_lowres',
            'kps_score_prefinal' if move_back_batch_to_time else 'kps_score',
            cfg.KRCNN.NUM_KEYPOINTS * time_dim,
            cfg.KRCNN.NUM_KEYPOINTS * time_dim,
            cfg.KRCNN.UP_SCALE)
        if move_back_batch_to_time:
            model.MoveTimeToBatchDimInverse(
                'kps_score_prefinal', 'kps_score_prefinal2', original_time_dim)
            model.MoveTimeToChannelDim(
                'kps_score_prefinal2', 'kps_score')

    return blob_out


def add_heatmap_losses(model, time_dim=1):
    # Reshape input from (N, K, H, W) to (NK, HW)
    model.net.Reshape(
        ['kps_score'], ['kps_score_reshaped', '_kps_score_old_shape'],
        shape=(-1, cfg.KRCNN.HEATMAP_SIZE * cfg.KRCNN.HEATMAP_SIZE))
    # Softmax across **space** (woahh....space!)
    kps_prob, loss_kps = model.net.SoftmaxWithLoss(
        ['kps_score_reshaped', 'keypoint_locations_int32', 'keypoint_weights'],
        ['kps_prob', 'loss_kps'],
        # DONOT scale the loss by time_dim! Somehow the values were same
        # for keypoints, whether I predict 17 or 51 keypoints.
        scale=cfg.KRCNN.LOSS_WEIGHT / cfg.NUM_GPUS,
        spatial=0)
    loss_gradients = blob_utils.get_loss_gradients(model, [loss_kps])
    model.losses = list(set(model.losses + ['loss_kps']))
    return loss_gradients


# ---------------------------------------------------------------------------- #
# Helper functions for building trainable models
# ---------------------------------------------------------------------------- #

def init_model(name, train, init_params=None):
    """
    Args:
        init_params: Set to true if you want to initialize the weights for sure
        (even if testing). Otherwise by default, it will initialize weights in
        train time and not in test time.
    """
    init_params = init_params or train
    return DetectionModelHelper(
        name=name, train=train, num_classes=cfg.MODEL.NUM_CLASSES,
        init_params=init_params)


def build_data_parallel_model(model, single_gpu_build_func):
    if model.train:
        all_loss_gradients = {}  # Will include loss gradients from all GPUs
        # Build the model on each GPU with correct name and device scoping
        for gpu_id in range(cfg.NUM_GPUS):
            with core.NameScope('gpu_{}'.format(gpu_id)):
                with core.DeviceScope(muji.OnGPU(gpu_id)):
                    all_loss_gradients.update(
                        single_gpu_build_func(model))
        # Add backward pass on all GPUs
        model.AddGradientOperators(all_loss_gradients)
        if cfg.NUM_GPUS > 1:
            # Need to all-reduce the per-GPU gradients if training with more
            # than 1 GPU
            all_params = model.TrainableParams()
            assert len(all_params) % cfg.NUM_GPUS == 0, \
                'This should not happen.'
            # The model parameters are replicated on each GPU, get the number
            # distinct parameter blobs (i.e., the number of parameter blobs on
            # each GPU)
            params_per_gpu = int(len(all_params) / cfg.NUM_GPUS)
            with core.DeviceScope(muji.OnGPU(cfg.ROOT_GPU_ID)):
                # Iterate over distinct parameter blobs
                for i in range(params_per_gpu):
                    # Gradients from all GPUs for this parameter blob
                    gradients = [
                        model.param_to_grad[p]
                        for p in all_params[i::params_per_gpu]
                    ]
                    if len(gradients) > 0:
                        if cfg.USE_NCCL:
                            model.net.NCCLAllreduce(gradients, gradients)
                        else:
                            muji.Allreduce(
                                model.net, gradients, reduced_affix='')
        for gpu_id in range(cfg.NUM_GPUS):
            # After all-reduce, all GPUs perform SGD updates on their identical
            # params and gradients in parallel
            add_parameter_update_ops(model, gpu_id)
    else:
        # Testing only supports running on a single GPU
        with core.NameScope('gpu_{}'.format(cfg.ROOT_GPU_ID)):
            with core.DeviceScope(muji.OnGPU(cfg.ROOT_GPU_ID)):
                single_gpu_build_func(model)


def add_parameter_update_ops(model, gpu_id):
    with core.DeviceScope(muji.OnGPU(gpu_id)):
        with core.NameScope('gpu_{}'.format(gpu_id)):
            # Learning rate of 0 is a dummy value to be set properly at the
            # start of training
            lr = model.param_init_net.ConstantFill(
                [], 'lr', shape=[1], value=0.0)
            one = model.param_init_net.ConstantFill(
                [], 'one', shape=[1], value=1.0)
            wd = model.param_init_net.ConstantFill(
                [], 'wd', shape=[1], value=cfg.SOLVER.WEIGHT_DECAY)

        for param in model.TrainableParams(gpu_id=gpu_id):
            logger.info('param ' + str(param) + ' will be updated')
            param_grad = model.param_to_grad[param]
            # Initialize momentum vector
            param_momentum = model.param_init_net.ConstantFill(
                [param], param + '_momentum', value=0.0)
            if param in model.biases:
                # Special treatment for biases (mainly to match historical impl.
                # details):
                # (1) Do not apply weight decay
                # (2) Use a 2x higher learning rate
                model.Scale(param_grad, param_grad, scale=2.0)
            elif cfg.SOLVER.WEIGHT_DECAY > 0:
                # Apply weight decay to non-bias weights
                model.WeightedSum([param_grad, one, param, wd], param_grad)
            # Update param_grad and param_momentum in place
            model.net.MomentumSGDUpdate(
                [param_grad, param_momentum, lr, param],
                [param_grad, param_momentum, param],
                momentum=cfg.SOLVER.MOMENTUM)


def blob_ref_to_list(blob_ref_or_list):
    if isinstance(blob_ref_or_list, core.BlobReference):
        return [blob_ref_or_list]
    return blob_ref_or_list


def get_suffix_net(name, prefix_ops, net, outputs):
    """Takes a list of ops (from a NetDef proto) which must be a prefix of the
    list of ops in the given net. It then returns a new net that contains only
    those ops in the net that are not in the prefix op list (i.e., a "suffix
    net").
    """
    assert list(prefix_ops) == net.Proto().op[:len(prefix_ops)], \
        'prefix_ops must be a prefix of net.Proto().op'
    for output in outputs:
        assert net.BlobIsDefined(output)
    new_net = net.Clone(name)

    del new_net.Proto().op[:]
    del new_net.Proto().external_input[:]
    del new_net.Proto().external_output[:]

    # Add suffix ops
    new_net.Proto().op.extend(net.Proto().op[len(prefix_ops):])
    # Add external input blobs
    # Treat any undefined blobs as external inputs
    input_names = [
        i for op in new_net.Proto().op for i in op.input
        if not new_net.BlobIsDefined(i)]
    new_net.Proto().external_input.extend(input_names)
    # Add external output blobs
    output_names = [str(o) for o in outputs]
    new_net.Proto().external_output.extend(output_names)
    return new_net, [new_net.GetBlobRef(o) for o in output_names]


def time_pool_blobs(blob_conv, model, body_head_link):
    if body_head_link == '':  # i.e., don't do any time pooling
        return blob_conv
    new_blob_conv = []
    was_list = False
    if isinstance(blob_conv, list):
        was_list = True
    for blob in blob_ref_to_list(blob_conv):
        if body_head_link == 'avg':
            new_blob = model.TimePool(blob, None, 'avg')
        elif body_head_link == 'slice-center':
            new_blob = model.SliceKeyFrame(blob, cfg.VIDEO.NUM_FRAMES_MID)
        else:
            raise NotImplementedError('Uknown body-head link {}'.format(
                body_head_link))
        new_blob_conv.append(new_blob)
    if not was_list:
        new_blob_conv = new_blob_conv[0]
    return new_blob_conv


# ---------------------------------------------------------------------------- #
# Hardcoded functions to create various types of common models
#
#            *** This type of model definition is deprecated ***
#            *** Use the generic composable versions instead ***
#
# ---------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------- #
# Fast R-CNN models
# ---------------------------------------------------------------------------- #

def VGG_CNN_M_1024_fast_rcnn(model):
    return build_generic_fast_rcnn_model(
        model,
        VGG_CNN_M_1024.add_VGG_CNN_M_1024_conv5_body,
        VGG_CNN_M_1024.add_VGG_CNN_M_1024_roi_fc_head)


def VGG16_fast_rcnn(model):
    return build_generic_fast_rcnn_model(
        model,
        VGG16.add_VGG16_conv5_body,
        VGG16.add_VGG16_roi_fc_head)


def ResNet50_fast_rcnn(model):
    return build_generic_fast_rcnn_model(
        model,
        ResNet.add_ResNet50_conv4_body,
        ResNet.add_ResNet_roi_conv5_head)


def ResNet101_fast_rcnn(model):
    return build_generic_fast_rcnn_model(
        model,
        ResNet.add_ResNet101_conv4_body,
        ResNet.add_ResNet_roi_conv5_head)


def ResNet50_fast_rcnn_frozen_features(model):
    return build_generic_fast_rcnn_model(
        model,
        ResNet.add_ResNet50_conv4_body,
        ResNet.add_ResNet_roi_conv5_head,
        freeze_conv_body=True)


def ResNet101_fast_rcnn_frozen_features(model):
    return build_generic_fast_rcnn_model(
        model,
        ResNet.add_ResNet101_conv4_body,
        ResNet.add_ResNet_roi_conv5_head,
        freeze_conv_body=True)


# ---------------------------------------------------------------------------- #
# RPN-only models
# ---------------------------------------------------------------------------- #

def VGG_CNN_M_1024_rpn(model):
    return build_generic_rpn_model(
        model,
        VGG_CNN_M_1024.add_VGG_CNN_M_1024_conv5_body)


def VGG16_rpn(model):
    return build_generic_rpn_model(
        model,
        VGG16.add_VGG16_conv5_body)


def ResNet50_rpn_conv4(model):
    return build_generic_rpn_model(
        model,
        ResNet.add_ResNet50_conv4_body)


def ResNet101_rpn_conv4(model):
    return build_generic_rpn_model(
        model,
        ResNet.add_ResNet101_conv4_body)


def VGG_CNN_M_1024_rpn_frozen_features(model):
    return build_generic_rpn_model(
        model,
        VGG_CNN_M_1024.add_VGG_CNN_M_1024_conv5_body,
        freeze_conv_body=True)


def VGG16_rpn_frozen_features(model):
    return build_generic_rpn_model(
        model,
        VGG16.add_VGG16_conv5_body,
        freeze_conv_body=True)


def ResNet50_rpn_conv4_frozen_features(model):
    return build_generic_rpn_model(
        model,
        ResNet.add_ResNet50_conv4_body,
        freeze_conv_body=True)


def ResNet101_rpn_conv4_frozen_features(model):
    return build_generic_rpn_model(
        model,
        ResNet.add_ResNet101_conv4_body,
        freeze_conv_body=True)


# ---------------------------------------------------------------------------- #
# Faster R-CNN models
# ---------------------------------------------------------------------------- #

def VGG16_faster_rcnn(model):
    assert cfg.MODEL.FASTER_RCNN
    return build_generic_fast_rcnn_model(
        model,
        VGG16.add_VGG16_conv5_body,
        VGG16.add_VGG16_roi_fc_head)


def ResNet50_faster_rcnn(model):
    assert cfg.MODEL.FASTER_RCNN
    return build_generic_fast_rcnn_model(
        model,
        ResNet.add_ResNet50_conv4_body,
        ResNet.add_ResNet_roi_conv5_head)


def ResNet101_faster_rcnn(model):
    assert cfg.MODEL.FASTER_RCNN
    return build_generic_fast_rcnn_model(
        model,
        ResNet.add_ResNet101_conv4_body,
        ResNet.add_ResNet_roi_conv5_head)


# ---------------------------------------------------------------------------- #
# R-FCN models
# ---------------------------------------------------------------------------- #

def VGG16_rfcn(model):
    return build_generic_rfcn_model(
        model,
        VGG16.add_VGG16_fcn)


def ResNet50_rfcn(model):
    return build_generic_rfcn_model(
        model,
        ResNet.add_ResNet50_conv5_body,
        dim_reduce=1024)


def ResNet101_rfcn(model):
    return build_generic_rfcn_model(
        model,
        ResNet.add_ResNet101_conv5_body,
        dim_reduce=1024)
