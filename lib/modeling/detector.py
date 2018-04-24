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
import os.path as osp
from caffe2.python import cnn, core, workspace, scope
from caffe2.proto import caffe2_pb2

from core.config import cfg
from ops.generate_proposals import GenerateProposalsOp
from ops.roi_blob_transforms import RoIToBatchFormatOp
from ops.debug_ops import PDBOp, TDBOp
from ops.generate_proposal_labels import GenerateProposalLabelsOp
from ops.collect_and_distribute_fpn_rpn_proposals import \
    CollectAndDistributeFpnRpnProposalsOp
import roi_data.fast_rcnn
from utils import lr_policy

import logging
logger = logging.getLogger(__name__)


class DetectionModelHelper(cnn.CNNModelHelper):
    def __init__(self, **kwargs):
        # Handle args specific to the DetectionModelHelper, others pass through
        # to CNNModelHelper
        self.train = kwargs.get('train', False)
        self.num_classes = kwargs.get('num_classes', -1)
        assert self.num_classes > 0, 'num_classes must be > 0'
        for k in ('train', 'num_classes'):
            if k in kwargs:
                del kwargs[k]
        kwargs['order'] = 'NCHW'
        # Defensively set cudnn_exhaustive_search to False in case the default
        # changes in CNNModelHelper. The detection code uses variable size
        # inputs that might not play nicely with cudnn_exhaustive_search.
        kwargs['cudnn_exhaustive_search'] = False
        super(DetectionModelHelper, self).__init__(**kwargs)
        self.roi_data_loader = None
        self.losses = []
        self.metrics = []
        self.do_not_update_params = []  # Param on this list are not updated
        self.net.Proto().type = cfg.MODEL.EXECUTION_TYPE
        self.net.Proto().num_workers = cfg.NUM_GPUS * 4
        self.prev_use_cudnn = self.use_cudnn

    def TrainableParams(self, gpu_id=-1):
        return [
            p for p in self.params
            if (
                p in self.param_to_grad and   # p has a gradient
                p not in self.do_not_update_params and  # not on the blacklist
                (gpu_id == -1 or  # filter for gpu assignment, if gpu_id set
                 str(p).find('gpu_{}'.format(gpu_id)) == 0)
            )]

    def AffineChannel(self, blob_in, blob_out, dim_out, share_with=None,
                      inplace=False):
        if cfg.MODEL.USE_BN:
            return self.SpatialBNLayer(blob_in, blob_out, dim_out, share_with,
                                       inplace)
        blob_out = blob_out or self.net.NextName()
        is_not_sharing = share_with is None
        param_prefix = blob_out if is_not_sharing else share_with
        scale = core.ScopedBlobReference(
            param_prefix + '_s', self.param_init_net)
        bias = core.ScopedBlobReference(
            param_prefix + '_b', self.param_init_net)
        if is_not_sharing:
            self.net.Proto().external_input.extend([str(scale), str(bias)])
            self.params.extend([scale, bias])
            self.weights.append(scale)
            self.biases.append(bias)
        if inplace:
            return self.net.AffineChannel([blob_in, scale, bias], blob_in)
        else:
            return self.net.AffineChannel([blob_in, scale, bias], blob_out)

    def AffineChannelNd(self, blob_in, blob_out, dim_out, share_with=None,
                        inplace=False):
        if cfg.MODEL.USE_BN:
            return self.SpatialBNLayer(blob_in, blob_out, dim_out, share_with,
                                       inplace)
        blob_out = blob_out or self.net.NextName()
        is_not_sharing = share_with is None
        param_prefix = blob_out if is_not_sharing else share_with
        scale = core.ScopedBlobReference(
            param_prefix + '_s', self.param_init_net)
        bias = core.ScopedBlobReference(
            param_prefix + '_b', self.param_init_net)
        if is_not_sharing:
            self.net.Proto().external_input.extend([str(scale), str(bias)])
            self.params.extend([scale, bias])
            self.weights.append(scale)
            self.biases.append(bias)
        if inplace:
            return self.net.AffineChannelNd([blob_in, scale, bias], blob_in)
        else:
            return self.net.AffineChannelNd([blob_in, scale, bias], blob_out)

    def SpatialBNLayer(self, blob_in, blob_out, dim_out,
                       share_with=None, inplace=False):
        if share_with is not None:
            raise NotImplementedError('Handle that')
        if inplace:
            logger.warning('Not supporting inplace yet (for {})'
                           .format(str(blob_in)))
            # blob_out = blob_in
        blob_out = blob_out or osp.basename(str(blob_out)) + '_bn'
        return self.SpatialBN(
            blob_in, blob_out, dim_out,
            epsilon=cfg.MODEL.BN_EPSILON,
            momentum=cfg.MODEL.BN_MOMENTUM,
            is_test=True if cfg.MODEL.USE_BN_TESTMODE_ONLY else not self.train)

    def RoIToBatchFormat(self, blob_in, blob_out):
        blob_out = blob_out or self.net.NextName()
        name = 'RoIToBatchFormatOp:' + str(blob_in)
        self.net.Python(RoIToBatchFormatOp().forward)(
            blob_in, blob_out, name=name)
        return blob_out

    def BreakPoint(self, blob_in, debug_type='TDB'):
        """ Add a debug break point layer.
            Args:
                blob_in (list of str/tensors): Blobs to investigate
                debug_type (str): Can be 'PDB'/'TDB'
        """
        name = 'BreakPoint:' + str(blob_in)
        Op = PDBOp if debug_type == 'PDB' else TDBOp
        self.net.Python(Op().forward)(
            blob_in, [], name=name)

    def GenerateProposals(
            self, blobs_in, blobs_out, anchors, spatial_scale):
        # blobs_in = ['rpn_cls_probs', 'rpn_bbox_pred', 'im_info']
        # blobs_out = ['rpn_rois', 'rpn_roi_probs']
        name = 'GenerateProposalsOp:' + ','.join([str(b) for b in blobs_in])
        self.net.Python(
            GenerateProposalsOp(anchors, spatial_scale, self.train).forward
        )(blobs_in, blobs_out, name=name)
        return blobs_out

    def GenerateProposalLabels(self, blobs_in):
        # blobs_in = ['rpn_rois', 'roidb', 'im_info']
        name = 'GenerateProposalLabelsOp:' + ','.join(
            [str(b) for b in blobs_in])

        # Get output blob names from the data loader
        blobs_out = roi_data.fast_rcnn.get_fast_rcnn_blob_names(
            is_training=self.train)
        blobs_out = [core.ScopedBlobReference(b) for b in blobs_out]

        self.net.Python(GenerateProposalLabelsOp().forward)(
            blobs_in, blobs_out, name=name)
        return blobs_out

    def CollectAndDistributeFpnRpnProposals(self):
        """Merges RPN proposals generated at various FPN levels and then
        redistributes those proposals to their appropriate FPN levels for use by
        the RoIFeatureTransform op.
        Input blobs: [rpn_rois_fpn<min>, ..., rpn_rois_fpn<max>,
                      rpn_roi_probs_fpn<min>, ..., rpn_roi_probs_fpn<max>]
        Output blobs: [rois_fpn<min>, ..., rois_rpn<max>, rois,
                       rois_idx_restore]
        If used during training, then the input blobs will also include
        [gt_boxes, roidb, im_info] and the output blobs will include (before
        rois) [labels, bbox_targets, bbox_inside_weights, bbox_outside_weights].
        """
        k_max = cfg.FPN.RPN_MAX_LEVEL
        k_min = cfg.FPN.RPN_MIN_LEVEL

        # Prepare input blobs
        rois_names = ['rpn_rois_fpn' + str(l) for l in range(k_min, k_max + 1)]
        score_names = [
            'rpn_roi_probs_fpn' + str(l) for l in range(k_min, k_max + 1)
        ]
        blobs_in = rois_names + score_names
        if self.train:
            blobs_in += ['roidb', 'im_info']
        blobs_in = [core.ScopedBlobReference(b) for b in blobs_in]
        name = 'CollectAndDistributeFpnRpnProposalsOp:' + ','.join(
            [str(b) for b in blobs_in]
        )

        # Prepare output blobs
        blobs_out = roi_data.fast_rcnn.get_fast_rcnn_blob_names(
            is_training=self.train)
        blobs_out = [core.ScopedBlobReference(b) for b in blobs_out]

        outputs = self.net.Python(
            CollectAndDistributeFpnRpnProposalsOp(self.train).forward
        )(blobs_in, blobs_out, name=name)

        return outputs

    def DropoutIfTraining(self, blob_in):
        """Add dropout to blob_in if the model is in training mode and
        cfg.TRAIN.DROPOUT is > 0."""
        blob_out = blob_in
        if self.train and cfg.TRAIN.DROPOUT > 0:
            blob_out = self.Dropout(
                blob_in, blob_in, ratio=cfg.TRAIN.DROPOUT, is_test=False)
        return blob_out

    def _do_roi_transform(self, method, blobs_in, blob_rois, blob_out,
                          bl_argmax, resolution, spatial_scale, sampling_ratio,
                          return_in_2d_format=False):
        """ Perform the RoIAlign/pool transforms, congnizant of the
        video cases.
        Args:
            keep_in_2d_format (bool): If you don't want to convert the 4D tensor
            back to 5D. This is useful for FPN case, when some levels don't have
            any RoIs and can't be transposed.
        """
        is_head_3d = False
        if cfg.MODEL.VIDEO_ON and cfg.VIDEO.BODY_HEAD_LINK == '':
            is_head_3d = True
        if is_head_3d:
            # Move the time dimension in rois
            blob_rois = self.RoIToBatchFormat(blob_rois, None)
            temporal_dim = self.GetTemporalDim(blobs_in)
            blobs_in = self.MoveTimeToBatchDim(blobs_in, None)
            # Use a temporary blob_out
            original_blob_out = blob_out
            blob_out = self.net.NextName()
        else:
            temporal_dim = 'zero'  # Just some dummy blob, won't be used
        # Now do the actual transform
        xform_out = self.net.__getattr__(method)(
            [blobs_in, blob_rois], [blob_out] + bl_argmax,
            pooled_w=resolution,
            pooled_h=resolution,
            spatial_scale=spatial_scale,
            sampling_ratio=sampling_ratio)
        if is_head_3d and not return_in_2d_format:
            # Need to keep_in_2d_format at times as if any dimension of the blob
            # is 0, the following does not work (as C2 Transpose op can't handle
            # blobs with any dimension 0). This happens in case of FPN
            # if modeling/FPN.py:map_rois_to_fpn_levels does not assign any
            # boxes to a certain level.
            xform_out = self.MoveTimeToBatchDimInverse(
                xform_out, original_blob_out, temporal_dim)
        return xform_out, temporal_dim, is_head_3d

    def RoIFeatureTransform(
            self, blobs_in, blob_out, blob_rois='rois', method='RoIPoolF',
            resolution=7, spatial_scale=1. / 16., sampling_ratio=0):
        """Add the specified RoI pooling method. Sampling_ratio is supported
        for some, but not all, methods."""
        assert method in (
            'RoIPoolF', 'RoIPoolLateQuantization', 'RoIWarp', 'RoIWarpMax',
            'RoIAlign'), 'Unknown pooling method: {}'.format(method)
        has_argmax = method in ('RoIPoolF', 'RoIPoolLateQuantization')
        if isinstance(blobs_in, list):
            # Pooling from multiple feature levels
            k_max = cfg.FPN.ROI_MAX_LEVEL  # coarsest level of pyramid
            k_min = cfg.FPN.ROI_MIN_LEVEL  # finest level of pyramid
            assert len(blobs_in) == k_max - k_min + 1
            bl_out_list = []
            for lvl in range(k_min, k_max + 1):
                bl_in = blobs_in[k_max - lvl]  # blobs_in is in reversed order
                sc = spatial_scale[k_max - lvl]  # in reversed order
                bl_rois = blob_rois + '_fpn' + str(lvl)
                bl_out = blob_out + '_fpn' + str(lvl)
                bl_argmax = ['_argmax_' + bl_out] if has_argmax else []
                bl_out, temporal_dim, is_head_3d = self._do_roi_transform(
                    method, bl_in, bl_rois, bl_out,
                    bl_argmax, resolution, sc, sampling_ratio,
                    return_in_2d_format=True)
                bl_out_list.append(bl_out)
            # Concat all pooled RoIs along batch dimension
            xform_shuffled, _ = self.net.Concat(
                bl_out_list,
                [blob_out + '_shuffled_bt', '_concat_' + blob_out],
                axis=0)
            if is_head_3d:
                xform_shuffled = self.MoveTimeToBatchDimInverse(
                    xform_shuffled, blob_out + '_shuffled', temporal_dim)
                # Move to channel for the batch permutation step
                xform_shuffled = self.MoveTimeToChannelDim(
                    xform_shuffled, None)
            # Unshuffle to match rois from dataloader
            restore_bl = blob_rois + '_idx_restore_int32'
            xform_out = self.net.BatchPermutation(
                [xform_shuffled, restore_bl], blob_out + '_tc')
            if is_head_3d:
                xform_out = self.MoveTimeToChannelDimInverse(
                    xform_out, blob_out, temporal_dim)
            else:
                blob_out = blob_out + '_tc'
        else:
            # Single feature level
            bl_argmax = ['_argmax_' + blob_out] if has_argmax else []
            # sampling_ratio is ignored for RoIPoolF and RoIPoolLateQuantization
            xform_out, _, _ = self._do_roi_transform(
                method, blobs_in, blob_rois, blob_out,
                bl_argmax, resolution, spatial_scale, sampling_ratio)
        # Only return the first blob (the transformed features)
        return xform_out

    def ConvShared(
            self, blob_in, blob_out, dim_in, dim_out, kernel,
            weight=None, bias=None, nd=False, **kwargs):
        use_bias = (
            False if ('no_bias' in kwargs and kwargs['no_bias']) else True)

        if self.use_cudnn:
            kwargs['engine'] = 'CUDNN'
            kwargs['exhaustive_search'] = self.cudnn_exhaustive_search
            if self.ws_nbytes_limit:
                kwargs['ws_nbytes_limit'] = self.ws_nbytes_limit

        if 'no_bias' in kwargs:
            del kwargs['no_bias']

        if nd:
            return self.ConvNd(
                blob_in,
                blob_out,
                dim_in, dim_out,
                kernel,
                weight=weight,
                bias=bias,
                **kwargs)
        else:
            if use_bias:
                blobs_in = [blob_in, weight, bias]
            else:
                blobs_in = [blob_in, weight]
            return self.net.Conv(
                blobs_in,
                blob_out,
                kernel=kernel,
                order=self.order,
                **kwargs)

    def BilinearInterpolation(
            self, blob_in, blob_out, dim_in, dim_out, up_scale):
        """Bilinear interpolation in space of scale
        Takes input of NxKxHxW and outputs NxKx(sH)x(sW), where s:= up_scale
        """
        assert dim_in == dim_out
        assert up_scale % 2 == 0, 'Scale should be even'

        def upsample_filt(size):
            factor = (size + 1) // 2
            if size % 2 == 1:
                center = factor - 1
            else:
                center = factor - 0.5
            og = np.ogrid[:size, :size]
            return ((1 - abs(og[0] - center) / factor) *
                    (1 - abs(og[1] - center) / factor))

        kernel_size = up_scale * 2
        bil_filt = upsample_filt(kernel_size)

        kernel = np.zeros(
            (dim_in, dim_out, kernel_size, kernel_size), dtype=np.float32)
        kernel[range(dim_out), range(dim_in), :, :] = bil_filt

        blob = self.ConvTranspose(
            blob_in, blob_out, dim_in, dim_out, kernel_size,
            stride=int(up_scale), pad=int(up_scale / 2),
            weight_init=('GivenTensorFill', {'values': kernel}),
            bias_init=('ConstantFill', {'value': 0.}))
        self.do_not_update_params.append(self.weights[-1])
        self.do_not_update_params.append(self.biases[-1])
        return blob

    def ConvAffine(  # args in the same order of Conv()
        self, blob_in, prefix, dim_in, dim_out, kernel, stride, pad,
        group=1, dilation=1,
        weight_init=None,
        bias_init=None,
        suffix='_bn',
        inplace=False
    ):
        """ConvAffine adds a Conv op followed by a AffineChannel op (which
        replaces BN during fine tuning).
        """
        conv_blob = self.Conv(
            blob_in,
            prefix,
            dim_in,
            dim_out,
            kernel,
            stride=stride,
            pad=pad,
            group=group,
            dilation=dilation,
            weight_init=weight_init,
            bias_init=bias_init,
            no_bias=1)
        blob_out = self.AffineChannel(
            conv_blob, prefix + suffix, dim_out, inplace=inplace)
        return blob_out

    def ConvAffineNd(  # args in the same order of Conv()
        self, blob_in, prefix, dim_in, dim_out, kernels, strides, pads,
        group=1, dilations=1,
        weight_init=None,
        bias_init=None,
        suffix='_bn',
        inplace=False
    ):
        """ConvAffineNd adds a Conv op followed by a AffineChannel op (which
        replaces BN during fine tuning).
        """
        conv_blob = self.ConvNd(
            blob_in,
            prefix,
            dim_in,
            dim_out,
            kernels,
            strides=strides,
            pads=pads,
            group=group,
            dilations=dilations,
            weight_init=weight_init,
            bias_init=bias_init,
            no_bias=1)
        blob_out = self.AffineChannelNd(
            conv_blob, prefix + suffix, dim_out, inplace=inplace)
        return blob_out

    # def Transpose(self, blob_in, blob_out, **kwargs):
    #     # The engine='DEFAULT' exists as cuDNN int32 transposes are broken with -1.
    #     # Gets fixed in D6043284. Can remove after rebasing onto that diff.
    #     kwargs['engine'] = 'DEFAULT'
    #     return self.net.Transpose(blob_in, blob_out, **kwargs)

    def GetValAtIdx(self, blob_in, idx):
        return self.Slice(blob_in, starts=[idx], ends=[idx + 1])

    def GetShapeDimIdx(self, blob_in, idx):
        shape_blob = self.Shape(blob_in)
        return self.GetValAtIdx(shape_blob, idx)

    def GetTemporalDim(self, blob_in):
        """
        blob_in must be 5 dim.
        TODO(rgirdhar): Add a assert to make sure of that.
        """
        temporal_dim = self.GetShapeDimIdx(blob_in, 2)
        return temporal_dim

    def BlobExists(self, blob):
        if blob is None:
            return False
        if self.net.BlobIsDefined(
                scope.CurrentNameScope() + osp.basename(str(blob))):
            return True
        return False

    def MoveTimeToBatchDim(self, blob_in, blob_out):
        blob_out = blob_out or osp.basename(str(blob_in)) + '_MovedTimeToBatchDim'
        inter_name = osp.basename(str(blob_in)) + '_MoveTimeToBatchDim_inter'
        if not self.BlobExists(inter_name + '1'):
            self.Transpose(blob_in, inter_name + '1', axes=(0, 2, 1, 3, 4))
        if not self.BlobExists(inter_name + '2'):
            self.Reshape(
                [inter_name + '1'], [inter_name + '2', inter_name + '3'],
                shape=(1, -1, 0, 0, 0))
        if not self.BlobExists(blob_out):
            self.Squeeze([inter_name + '2'], [blob_out], dims=[0])
        return blob_out

    def MoveTimeToChannelDim(self, blob_in, blob_out):
        blob_out = blob_out or osp.basename(str(blob_in)) + '_MovedTimeToChDim'
        inter_name = osp.basename(str(blob_in)) + '_MoveTimeToChannelDim_inter'
        if not self.BlobExists(inter_name + '0'):
            self.Transpose(blob_in, inter_name + '0', axes=(0, 2, 1, 3, 4))
        if not self.BlobExists(inter_name + '1'):
            self.Reshape(
                [inter_name + '0'], [inter_name + '1', inter_name + '2'],
                shape=(0, 1, -1, 0, 0))
        if not self.BlobExists(blob_out):
            self.Squeeze([inter_name + '1'], [blob_out], dims=[1])
        return blob_out

    def MoveTimeToBatchIfNotAlreadyDone(self, blob_name):
        """ Wrapper around MoveTimeToBatch, avoids moving blobs
        that already have a moved version."""
        blob_out_name = blob_name + '_bt'
        if not self.BlobExists(blob_out_name):
            self.MoveTimeToBatchDim(blob_name, blob_out_name)
        return blob_out_name

    def GetNewShape(self, *args):
        new_shape = self.net.NextName()
        existing_blobs = {0: 'zero', -1: 'minus1'}
        shape_blob = []
        for arg in args:
            if arg in existing_blobs:
                shape_blob.append(existing_blobs[arg])
            else:
                shape_blob.append(arg)
        self.net.Concat(shape_blob, [new_shape, self.net.NextName()], axis=0)
        return new_shape

    def MoveTimeToBatchDimInverse(self, blob_in, blob_out, temporal_dim):
        """
        Args:
            new_shape (tensor): Is the output from the MoveTimeToBatchDim op.
            If not provided it'll try to do something smart, but won't work if
            the time stride is > 1
        """
        blob_out = blob_out or osp.basename(str(blob_in)) + '_MovedTimeToBatchDimInv'
        inter_name = osp.basename(str(blob_in)) + '_MoveTimeToBatchDimInverse_inter'
        if not self.BlobExists(inter_name + '0'):
            self.ExpandDims([blob_in], [inter_name + '0'], dims=[0])
        new_shape = self.net.NextName()
        # TODO(rgirdhar): Move this to GetNewShape
        self.net.Concat(
            ['minus1', temporal_dim, 'zero', 'zero', 'zero'],
            [new_shape, self.net.NextName()], axis=0)
        if not self.BlobExists(inter_name + '1'):
            self.Reshape([inter_name + '0', new_shape],
                         [inter_name + '1', self.net.NextName()])
        if not self.BlobExists(blob_out):
            self.Transpose(inter_name + '1', blob_out, axes=(0, 2, 1, 3, 4))
        return blob_out

    def MoveTimeToChannelDimInverse(self, blob_in, blob_out, temporal_dim):
        """
        Args:
            new_shape (tensor): Is the output from the MoveTimeToBatchDim op.
            If not provided it'll try to do something smart, but won't work if
            the time stride is > 1
        """
        blob_out = blob_out or osp.basename(str(blob_in)) + '_MovedTimeToChDimInv'
        inter_name = osp.basename(str(blob_in)) + '_MoveTimeToChannelDimInverse_inter'
        if not self.BlobExists(inter_name + '0'):
            self.ExpandDims([blob_in], [inter_name + '0'], dims=[1])
        new_shape = self.net.NextName()
        # TODO(rgirdhar): Move this to GetNewShape
        self.net.Concat(
            ['zero', temporal_dim, 'minus1', 'zero', 'zero'],
            [new_shape, self.net.NextName()], axis=0)
        if not self.BlobExists(blob_out):
            self.Reshape([inter_name + '0', new_shape],
                         [inter_name + '1', self.net.NextName()])
        if not self.BlobExists(blob_out):
            self.Transpose(inter_name + '1', blob_out, axes=(0, 2, 1, 3, 4))
        return blob_out

    def TimePool(self, blob_in, blob_out, pool_type):
        blob_out = blob_out or osp.basename(str(blob_in)) + \
            '_TimePooled_{}'.format(pool_type)
        if pool_type == 'center-slice':
            raise NotImplementedError()
        elif pool_type == 'avg':
            blob_trans = self.net.NextName()
            self.Transpose(blob_in, blob_trans, axes=(0, 1, 3, 4, 2))
            return self.ReduceBackMean(blob_trans, blob_out)
        else:
            raise NotImplementedError('Unknown type {}'.format(pool_type))

    def SliceKeyFrame(self, blob_in, N):
        keyframe = int(N / 2)
        out = self.Slice([blob_in], [blob_in + '_slicekey'],
                         starts=[0, 0, keyframe, 0, 0],
                         ends=[-1, -1, keyframe + 1, -1, -1])
        return self.Squeeze(out, dims=[2])

    def DisableCudnn(self):
        self.prev_use_cudnn = self.use_cudnn
        self.use_cudnn = False

    def RestorePreviousUseCudnn(self):
        prev_use_cudnn = self.use_cudnn
        self.use_cudnn = self.prev_use_cudnn
        self.prev_use_cudnn = prev_use_cudnn

    def UpdateWorkspaceLr(self, cur_iter):
        """Updates the model's current learning rate and the workspace (learning
        rate and update history/momentum blobs).
        """
        # The workspace is the one source of truth for the lr
        # The lr is always the same on all GPUs
        cur_lr = workspace.FetchBlob('gpu_0/lr')[0]
        new_lr = lr_policy.get_lr_at_iter(cur_iter)
        # There are no type conversions between the lr in Python and the lr in
        # the GPU (both are float32), so exact comparision is ok
        if cur_lr != new_lr:
            ratio = np.max((new_lr / cur_lr, cur_lr / new_lr))
            if ratio > cfg.SOLVER.LOG_LR_CHANGE_THRESHOLD:
                logger.info(
                    'Changing learning rate {:.6f} -> {:.6f} at iter {:d}'.
                    format(cur_lr, new_lr, cur_iter))
            self._SetNewLr(cur_lr, new_lr)
        return new_lr

    def _SetNewLr(self, cur_lr, new_lr):
        """Do the actual work of updating the model and workspace blobs.
        """
        for i in range(cfg.NUM_GPUS):
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, i)):
                workspace.FeedBlob(
                    'gpu_{}/lr'.format(i), np.array([new_lr], dtype=np.float32))
        ratio = np.max((new_lr / cur_lr, cur_lr / new_lr))
        if cfg.SOLVER.SCALE_MOMENTUM and cur_lr > 1e-7 and \
                ratio > cfg.SOLVER.SCALE_MOMENTUM_THRESHOLD:
            self._CorrectMomentum(new_lr / cur_lr)

    def _CorrectMomentum(self, correction):
        """The MomentumSGDUpdate op implements the update V as

            V := mu * V + lr * grad,

        where mu is the momentum factor, lr is the learning rate, and grad is the
        stochastic gradient. Since V is not defined independently of the learning
        rate (as it should ideally be), when the learning rate is changed we should
        scale the update history V in order to make it compatible in scale with
        lr * grad.
        """
        logger.info(
            'Scaling update history by {:.6f} (new lr / old lr)'.
            format(correction))
        for i in range(cfg.NUM_GPUS):
            with core.DeviceScope(core.DeviceOption(caffe2_pb2.CUDA, i)):
                for param in self.TrainableParams(gpu_id=i):
                    op = core.CreateOperator(
                        'Scale', [param + '_momentum'], [param + '_momentum'],
                        scale=correction)
                    workspace.RunOperatorOnce(op)

    def PrintShape(self, tensor):
        self.Print(self.Shape(tensor, osp.basename(str(tensor)) + '_shape'), [])
