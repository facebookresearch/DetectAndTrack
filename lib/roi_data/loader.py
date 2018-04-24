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
from collections import OrderedDict
import threading
import multiprocessing
import uuid
import signal
import time

from core.config import cfg
from roi_data.minibatch import get_minibatch, get_minibatch_blob_names
from utils.coordinator import Coordinator, coordinated_get, coordinated_put

from caffe2.proto import caffe2_pb2
from caffe2.python import core, workspace

import logging
logger = logging.getLogger(__name__)


class RoIDataLoader(object):
    def __init__(
            self,
            roidb,
            num_workers=4,
            num_enqueuers=1,
            minibatch_queue_size=64,
            blobs_queue_capacity=8):
        self._roidb = roidb
        self._lock = multiprocessing.Lock()
        self._perm = np.arange(len(self._roidb))
        self._cur = 0  # _perm cursor
        # The minibatch queue holds prepared training data in host (CPU) memory
        # When training with N > 1 GPUs, each element in the minibatch queue
        # is actually a partial minibatch which contributes 1 / N examples to
        # the overall minibatch
        self._manager = multiprocessing.Manager()
        # Using a multiprocessing.manager.Queue instead of
        # multiprocessing.Queue, because the latter hangs when
        # exitting during worker.join. Got this idea
        # from https://stackoverflow.com/a/33153048
        self._minibatch_queue = self._manager.Queue(maxsize=minibatch_queue_size)
        self._minibatch_queue_maxsize = minibatch_queue_size
        self._blobs_queue_capacity = blobs_queue_capacity
        # Random identificator to deal with multiple instances of RoIDataLoaders
        self._loader_id = uuid.uuid4()
        self._blobs_queue_name = 'roi_blobs_queue_{}'.format(self._loader_id)
        # "worker" threads construct (partial) minibatches and put them on the
        # minibatch queue
        self._num_workers = num_workers
        # "enqueuer" threads get (partial) minibatches from the minibatch queue
        # and enqueue them on GPU blob queues
        self._num_enqueuers = num_enqueuers
        self._num_gpus = cfg.NUM_GPUS
        self.coordinator = Coordinator()

        self._output_names = get_minibatch_blob_names()
        self._perm, self._cur = self._shuffle_roidb_inds(self._roidb)

        # FOLLOWING is the stuff needed by the MULTIPROCESSING module;
        # Keeping it in init so that I can run the minibatch_loader2 in
        # debug mode from train_net.py
        # Previous comments:
        # The variables can not be shared as the class, so need to share
        # through manager to work with multiprocessing
        # manager = multiprocessing.Manager()
        # manager is SLOWW!!! so, using a normal dict and read-write vars
        # separately. Note that the following dict is NOT SHARED. A copy
        # will exist in each worker, so each worker can read it, but any
        # modifications will also be local. This is fine because I'll only
        # add READ_ONLY objects into this dict.
        self.shared_readonly_dict = {}
        # No need to synchronize the following things, since they are never
        # modified in the processes (only accessed), so a simple dictionary
        # is okay.
        self.shared_readonly_dict['output_names'] = self.get_output_names()
        self.shared_readonly_dict['roidb'] = self._roidb
        # Following will be modified, but always within a self._lock;
        # no a non-locking sync-ed variable would be good enough.
        self.mp_cur = multiprocessing.Value('i', self._cur, lock=False)
        self.mp_perm = multiprocessing.Array('i', self._perm.tolist(), lock=False)

        self.create_threads()

    @staticmethod
    def _shuffle_roidb_inds(roidb):
        """Randomly permute the training roidb. Not thread safe."""
        if cfg.TRAIN.ASPECT_GROUPING:
            widths = np.array([r['width'] for r in roidb])
            heights = np.array([r['height'] for r in roidb])
            horz = (widths >= heights)
            vert = np.logical_not(horz)
            horz_inds = np.where(horz)[0]
            vert_inds = np.where(vert)[0]
            inds = np.hstack((
                np.random.permutation(horz_inds),
                np.random.permutation(vert_inds)))
            inds = np.reshape(inds, (-1, 2))
            row_perm = np.random.permutation(np.arange(inds.shape[0]))
            inds = np.reshape(inds[row_perm, :], (-1,))
            perm = inds
        else:
            perm = np.random.permutation(np.arange(len(roidb)))
        cur = 0
        return perm, cur

    @staticmethod
    def _get_next_minibatch_inds(shared_readonly_dict, lock, mp_cur, mp_perm):
        """Return the roidb indices for the next minibatch. Thread safe."""
        with lock:
            roidb = shared_readonly_dict['roidb']
            if mp_cur.value + cfg.TRAIN.IMS_PER_BATCH >= len(roidb):
                mp_perm, mp_cur.value = \
                    RoIDataLoader._shuffle_roidb_inds(roidb)

            db_inds = np.array([mp_perm[i] for i in range(
                mp_cur.value, mp_cur.value + cfg.TRAIN.IMS_PER_BATCH)])
            if 0:  # debug to make sure different processes are reading diff lists
                logger.info('{} is reading {}'.format(
                    multiprocessing.current_process().name,
                    db_inds))
            mp_cur.value += cfg.TRAIN.IMS_PER_BATCH
        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch. DEPRECATED.
        This only exists for debugging (in train_net.py) and for
        benchmarking."""
        roidb = self._roidb
        valid = False
        while not valid:
            db_inds = self._get_next_minibatch_inds(
                {'roidb': roidb}, self._lock,
                multiprocessing.Value('i', self._cur, lock=False),
                self._perm)
            minibatch_db = [roidb[i] for i in db_inds]
            blobs, valid = get_minibatch(minibatch_db)
        return blobs

    @staticmethod
    def _get_next_minibatch2(shared_readonly_dict, lock, mp_cur, mp_perm):
        """Return the blobs to be used for the next minibatch. Thread safe."""
        roidb = shared_readonly_dict['roidb']
        valid = False
        while not valid:
            db_inds = RoIDataLoader._get_next_minibatch_inds(
                shared_readonly_dict, lock, mp_cur, mp_perm)
            minibatch_db = [roidb[i] for i in db_inds]
            blobs, valid = get_minibatch(minibatch_db)
        return blobs

    def get_output_names(self):
        return self._output_names

    def create_blobs_queue(self):
        """Create a BlobsQueue in the workspace to hold mini-batches."""
        workspace.RunOperatorOnce(
            core.CreateOperator(
                "CreateBlobsQueue",
                [], [self._blobs_queue_name],
                num_blobs=len(self.get_output_names()),
                capacity=self._blobs_queue_capacity))

    def close_blobs_queue(self):
        """Close a BlobsQueue."""
        workspace.RunOperatorOnce(
            core.CreateOperator(
                "CloseBlobsQueue",
                [self._blobs_queue_name], []))

    def minibatch_loader(self):
        """Load mini-batches and put them onto the mini-batch queue."""
        """This function is now DEPRECATED, won't work with multiprocessing. """
        with self.coordinator.stop_on_exception():
            while not self.coordinator.should_stop():
                blobs = self._get_next_minibatch()
                # Blobs must be queued in the order specified by
                # self.get_output_names
                ordered_blobs = OrderedDict()
                for key in self.get_output_names():
                    assert blobs[key].dtype in (np.int32, np.float32), \
                        'Blob {} of dtype {} must have dtype of ' \
                        'np.int32 or np.float32'.format(key, blobs[key].dtype)
                    ordered_blobs[key] = blobs[key]
                coordinated_put(
                    self.coordinator, self._minibatch_queue, ordered_blobs)
        logger.info('Stopping mini-batch loading thread')

    @staticmethod
    def minibatch_loader2(shared_readonly_dict, minibatch_queue, lock,
                          mp_cur, mp_perm, coordinator):
        """Load mini-batches and put them onto the mini-batch queue."""
        output_names = shared_readonly_dict['output_names']
        with coordinator.stop_on_exception():
            while not coordinator.should_stop():
                blobs = RoIDataLoader._get_next_minibatch2(
                    shared_readonly_dict, lock, mp_cur, mp_perm)
                # Blobs must be queued in the order specified by
                # self.get_output_names
                ordered_blobs = OrderedDict()
                for key in output_names:
                    assert blobs[key].dtype in (np.int32, np.float32), \
                        'Blob {} of dtype {} must have dtype of ' \
                        'np.int32 or np.float32'.format(key, blobs[key].dtype)
                    ordered_blobs[key] = blobs[key]
                coordinated_put(
                    coordinator, minibatch_queue, ordered_blobs)
        logger.info('Stopping mini-batch loading thread')

    def enqueue_blobs(self, gpu_id, blob_names, blobs):
        """Put a mini-batch on a BlobsQueue."""
        assert len(blob_names) == len(blobs)
        t = time.time()
        blob_names = [
            'gpu_{}/{}'.format(gpu_id, blob_name) for blob_name in blob_names]
        dev = core.DeviceOption(caffe2_pb2.CUDA, gpu_id)
        for (blob_name, blob) in zip(blob_names, blobs):
            workspace.FeedBlob(blob_name, blob, device_option=dev)
        logger.debug('enqueue_blobs {}: workspace.FeedBlob: {}'.
                     format(gpu_id, time.time() - t))
        t = time.time()
        op = core.CreateOperator(
            "EnqueueBlobs",
            ['gpu_{}/{}'.format(gpu_id, self._blobs_queue_name)] + blob_names,
            blob_names, device_option=dev)
        workspace.RunOperatorOnce(op)
        logger.debug('enqueue_blobs {}: workspace.RunOperatorOnce: {}'.
                     format(gpu_id, time.time() - t))

    def enqueue_blobs_thread(self, gpu_id, blob_names):
        """Transfer mini-batches from a mini-batch queue to a BlobsQueue."""
        with self.coordinator.stop_on_exception():
            while not self.coordinator.should_stop():
                if self._minibatch_queue.qsize == 0:
                    logger.warning('Mini-batch queue is empty')
                blobs = coordinated_get(
                    self.coordinator, self._minibatch_queue)
                self.enqueue_blobs(
                    gpu_id,
                    blob_names,
                    blobs.values())
                logger.debug(
                    'batch queue size {}'.format(self._minibatch_queue.qsize()))
            logger.info('Stopping enqueue thread')

    def create_threads(self):
        # Create mini-batch loader threads, each of which builds mini-batches
        # and places them into a queue in CPU memory
        threading_fn = multiprocessing.Process
        self._workers = [
            threading_fn(target=RoIDataLoader.minibatch_loader2,
                         args=(self.shared_readonly_dict,
                               self._minibatch_queue, self._lock,
                               self.mp_cur, self.mp_perm, self.coordinator))
            for _ in range(self._num_workers)
        ]

        # Create one BlobsQueue per GPU, each of which feeds a blob in GPU
        # memory to a net
        for gpu_id in range(self._num_gpus):
            with core.NameScope('gpu_{}'.format(gpu_id)):
                self.create_blobs_queue()

        # An enqueuer thread moves mini-batches from the shared CPU memory queue
        # to a GPU blobs queue
        # Each GPU will have it's own pool of enqueuer threads
        # Create one blob for each
        # (loader output, enqueuer thread, RoIDataLoader instance) triple:
        #   <loader_output>_enqueue_<enqueuer_thread_id>_<loader_id>
        blob_names = self.get_output_names()
        enqueue_blob_names = [
            ['{}_enqueue_{}_{}'.format(blob_name, i, self._loader_id)
                for blob_name in blob_names]
            for i in range(self._num_enqueuers)
        ]
        for gpu_id in range(self._num_gpus):
            with core.NameScope('gpu_{}'.format(gpu_id)):
                with core.DeviceScope(
                        core.DeviceOption(caffe2_pb2.CUDA, gpu_id)):
                    for blob_list in enqueue_blob_names:
                        for blob in blob_list:
                            workspace.CreateBlob(core.ScopedName(blob))
        # Create enqueuer threads
        self._enqueuers = [
            # This is enqueueing into C2, can't be done by multiple processes
            # so needs to be done using threading module
            threading.Thread(
                target=self.enqueue_blobs_thread,
                args=(gpu_id, enqueue_blob_names[i]))
            for gpu_id in range(self._num_gpus)
            for i in range(self._num_enqueuers)]

    def start(self, prefill=False):
        for w in self._workers + self._enqueuers:
            w.start()
        if prefill:
            logger.info('Pre-filling mini-batch queue...')
            while not self._minibatch_queue.full():
                logger.info('  [{:d}/{:d}]'.
                            format(self._minibatch_queue.qsize(),
                                   self._minibatch_queue_maxsize))
                time.sleep(0.1)
                # Detect failure and shutdown
                if self.coordinator.should_stop():
                    self.shutdown()
                    break

    def join(self):
        logger.info('Join-ing all worker threads...')
        for w in self._workers + self._enqueuers:
            logger.info('Join-ing {}'.format(w))
            w.join()  # add a timeout, just in case

    def shutdown(self):
        self.coordinator.request_stop()
        self.coordinator.wait_for_stop()
        for i in range(self._num_gpus):
            with core.NameScope('gpu_{}'.format(i)):
                self.close_blobs_queue()
        self.join()

    def register_sigint_handler(self):
        def signal_handler(signal, frame):
            logger.info(
                'SIGINT: Shutting down RoIDataLoader threads and exiting...')
            self.shutdown()
        signal.signal(signal.SIGINT, signal_handler)
