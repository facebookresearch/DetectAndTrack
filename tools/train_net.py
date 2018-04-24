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
import cv2  # NOQA (Must import before importing caffe2 due to bug in cv2)

from caffe2.python import workspace, utils as c2_utils
from core.config import (
    cfg, cfg_from_list, cfg_from_file, get_output_dir, assert_and_infer_cfg)
from datasets.roidb import combined_roidb_for_training
from modeling import model_builder
import utils.net as nu
import utils.c2
from utils.timer import Timer
from utils.logger import log_json_stats, SmoothedValue
import test_net
import argparse
import pprint
import numpy as np
import sys
import os
import datetime
import logging

utils.c2.import_detectron_ops()

FORMAT = '%(levelname)s %(filename)s:%(lineno)4d: %(message)s'
logging.basicConfig(level=logging.INFO, format=FORMAT, stream=sys.stdout)
logger = logging.getLogger(__name__)

WIN_SZ = 20      # Median filter window size for smoothing logged values
LOG_PERIOD = 20  # Output logging period in SGD iterations


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detection network')
    parser.add_argument(
        '--skip-test', dest='skip_test', help='do not test the final model',
        action='store_true')
    parser.add_argument(
        '--cfg', dest='cfg_file', help='optional config file', default=None,
        type=str)
    parser.add_argument(
        '--multi-gpu-testing', dest='multi_gpu_testing',
        help='using cfg.NUM_GPUS for inference', action='store_true')
    parser.add_argument(
        'opts', help='See lib/core/config.py for all options', default=None,
        nargs=argparse.REMAINDER)
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def set_loggers():
    logging.getLogger('roi_data.loader').setLevel(logging.INFO)


def optimize_memory(model):
    from caffe2.python import memonger
    for device in range(cfg.NUM_GPUS):
        namescope = 'gpu_{}/'.format(device)
        losses = [namescope + l for l in model.losses]
        model.net._net = memonger.share_grad_blobs(
            model.net,
            losses,
            set(model.param_to_grad.values()),
            namescope,
            share_activations=cfg.MEMONGER_SHARE_ACTIVATIONS)


def create_model():
    start_iter = 0
    if cfg.CLUSTER.ON_CLUSTER and cfg.CLUSTER.AUTO_RESUME:
        import re
        output_dir = get_output_dir(training=True)
        final_path = os.path.join(output_dir, 'model_final.pkl')
        if os.path.exists(final_path):
            logger.info('model_final.pkl exists; no need to train!')
            return None, None, {'final': final_path}

        files = os.listdir(output_dir)
        for f in files:
            iter_string = re.findall(r'(?<=model_iter)\d+(?=\.pkl)', f)
            if len(iter_string) > 0:
                checkpoint_iter = int(iter_string[0])
                if checkpoint_iter > start_iter:
                    # Start one iteration immediately after the checkpoint iter
                    start_iter = checkpoint_iter + 1
                    resume_weights_file = f

        if start_iter > 0:
            cfg.TRAIN.WEIGHTS = os.path.join(output_dir, resume_weights_file)
            logger.info(
                '========> Resuming from checkpoint {} with start iter {}'.
                format(cfg.TRAIN.WEIGHTS, start_iter))

    logger.info('Building nework: {}'.format(cfg.MODEL.TYPE))
    model = model_builder.create(cfg.MODEL.TYPE, train=True)
    if cfg.MEMONGER:
        optimize_memory(model)
    workspace.RunNetOnce(model.param_init_net)
    return model, start_iter, {}


def add_model_inputs(model):
    logger.info('Loading dataset: {}'.format(cfg.TRAIN.DATASET))
    roidb = combined_roidb_for_training(
        cfg.TRAIN.DATASET, cfg.TRAIN.PROPOSAL_FILE)
    logger.info('{:d} roidb entries'.format(len(roidb)))
    model_builder.add_inputs(model, roidb=roidb)


def dump_proto_files(model, output_dir):
    with open(os.path.join(output_dir, 'net.pbtxt'), 'w') as fid:
        fid.write(str(model.net.Proto()))
    with open(os.path.join(output_dir, 'param_init_net.pbtxt'), 'w') as fid:
        fid.write(str(model.param_init_net.Proto()))


def net_trainer():
    model, start_iter, checkpoints = create_model()
    if 'final' in checkpoints:
        return checkpoints

    add_model_inputs(model)

    if cfg.TRAIN.WEIGHTS:
        nu.initialize_gpu_0_from_weights_file(model, cfg.TRAIN.WEIGHTS)
    # Even if we're randomly initializing we still need to synchronize
    # parameters across GPUs
    nu.broadcast_parameters(model)
    workspace.CreateNet(model.net)

    output_dir = get_output_dir(training=True)
    logger.info('Outputs saved to: {:s}'.format(os.path.abspath(output_dir)))
    dump_proto_files(model, output_dir)
    json_out_file = os.path.join(output_dir, 'json_stats.log')

    # Start loading mini-batches and enqueuing blobs
    model.roi_data_loader.register_sigint_handler()
    # DEBUG data loading
    if cfg.DEBUG.DATA_LOADING:
        for _ in range(10000000):
            # this was with threading...
            # model.roi_data_loader._get_next_minibatch()
            model.roi_data_loader._get_next_minibatch2(
                model.roi_data_loader.shared_readonly_dict,
                model.roi_data_loader._lock,
                model.roi_data_loader.mp_cur,
                model.roi_data_loader.mp_perm)
        sys.exit(0)
    model.roi_data_loader.start(prefill=True)

    smoothed_values = {
        key: SmoothedValue(WIN_SZ) for key in model.losses + model.metrics}
    iter_values = {key: 0 for key in model.losses + model.metrics}
    total_loss = SmoothedValue(WIN_SZ)
    iter_time = SmoothedValue(WIN_SZ)
    mb_qsize = SmoothedValue(WIN_SZ)
    iter_timer = Timer()
    checkpoints = {}
    for i in range(start_iter, cfg.SOLVER.MAX_ITER):
        iter_timer.tic()
        lr = model.UpdateWorkspaceLr(i)
        workspace.RunNet(model.net.Proto().name)
        if i == start_iter:
            nu.print_net(model)
        iter_time.AddValue(iter_timer.toc(average=False))
        for k in iter_values.keys():
            if k in model.losses:
                iter_values[k] = nu.sum_multi_gpu_blob(k)
            else:
                iter_values[k] = nu.average_multi_gpu_blob(k)
        for k, v in smoothed_values.items():
            v.AddValue(iter_values[k])
        loss = np.sum(np.array([iter_values[k] for k in model.losses]))
        total_loss.AddValue(loss)
        mb_qsize.AddValue(model.roi_data_loader._minibatch_queue.qsize())

        if i % LOG_PERIOD == 0 or i == cfg.SOLVER.MAX_ITER - 1:
            eta_seconds = iter_timer.average_time * (cfg.SOLVER.MAX_ITER - i)
            eta = str(datetime.timedelta(seconds=int(eta_seconds)))
            mem_stats = c2_utils.GetGPUMemoryUsageStats()
            mem_usage = np.max(mem_stats['max_by_gpu'][:cfg.NUM_GPUS])
            stats = dict(
                iter=i,
                lr=float(lr),
                time=iter_timer.average_time,
                loss=total_loss.GetMedianValue(),
                eta=eta,
                mb_qsize=int(np.round(mb_qsize.GetMedianValue())),
                mem=int(np.ceil(mem_usage / 1024 / 1024)))
            for k, v in smoothed_values.items():
                stats[k] = v.GetMedianValue()
            log_json_stats(stats, json_out_file=json_out_file)
        if cfg.DEBUG.STOP_TRAIN_ITER:
            import pdb
            pdb.set_trace()

        if ((i + 1) % int(cfg.TRAIN.SNAPSHOT_ITERS / cfg.NUM_GPUS) == 0 and
                i > start_iter):
            checkpoints[i] = os.path.join(
                output_dir, 'model_iter{}.pkl'.format(i))
            nu.save_model_to_weights_file(checkpoints[i], model)

        if i == start_iter + LOG_PERIOD:
            # Reset the iter timer after the first LOG_PERIOD iterations to
            # discard initial iterations that have outlier timings
            iter_timer.reset()

        if np.isnan(loss):
            logger.critical('Loss is NaN, exiting...')
            os._exit(0)  # FB: use code 0 to avoid flow retries

    # Save the final model
    checkpoints['final'] = os.path.join(output_dir, 'model_final.pkl')
    nu.save_model_to_weights_file(checkpoints['final'], model)
    # Shutdown data loading threads
    model.roi_data_loader.shutdown()
    return checkpoints


def test_model(model_file, multi_gpu_testing, opts=None):
    # All arguments to inference functions are passed via cfg
    cfg.TEST.WEIGHTS = model_file
    # Clear memory before inference
    workspace.ResetWorkspace()
    # Run inference
    test_net.main(multi_gpu_testing=multi_gpu_testing)


if __name__ == '__main__':
    workspace.GlobalInit(
        ['caffe2', '--caffe2_log_level=0', '--caffe2_gpu_memory_tracking'])
    set_loggers()
    # TODO(rbg): set C2 random seed
    np.random.seed(cfg.RNG_SEED)
    args = parse_args()
    logger.info('Called with args:')
    logger.info(args)
    if args.cfg_file is not None:
        cfg_from_file(args.cfg_file)
    if args.opts is not None:
        cfg_from_list(args.opts)
    assert_and_infer_cfg()
    logger.info('Training with config:')
    logger.info(pprint.pformat(cfg))
    checkpoints = net_trainer()
    if not args.skip_test:
        test_model(checkpoints['final'], args.multi_gpu_testing, args.opts)
    if cfg.FINAL_MSG != '':
        logger.info(cfg.FINAL_MSG)
