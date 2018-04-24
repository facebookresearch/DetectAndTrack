from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np

from datasets.posetrack.poseval.py.evaluateAP import evaluateAP
from datasets.posetrack.poseval.py.evaluateTracking import evaluateTracking
from datasets.posetrack.poseval.py.eval_helpers import (
    Joint, printTable, load_data_dir)


def evaluate(gtdir, preddir, eval_pose=True, eval_track=True,
             eval_upper_bound=False):
    gtFramesAll, prFramesAll = load_data_dir(['', gtdir, preddir])

    print('# gt frames  :', len(gtFramesAll))
    print('# pred frames:', len(prFramesAll))

    apAll = np.full((Joint().count + 1, 1), np.nan)
    preAll = np.full((Joint().count + 1, 1), np.nan)
    recAll = np.full((Joint().count + 1, 1), np.nan)
    if eval_pose:
        apAll, preAll, recAll = evaluateAP(gtFramesAll, prFramesAll)
        print('Average Precision (AP) metric:')
        printTable(apAll)

    metrics = np.full((Joint().count + 4, 1), np.nan)
    if eval_track:
        metricsAll = evaluateTracking(
            gtFramesAll, prFramesAll, eval_upper_bound)

        for i in range(Joint().count + 1):
            metrics[i, 0] = metricsAll['mota'][0, i]
        metrics[Joint().count + 1, 0] = metricsAll['motp'][0, Joint().count]
        metrics[Joint().count + 2, 0] = metricsAll['pre'][0, Joint().count]
        metrics[Joint().count + 3, 0] = metricsAll['rec'][0, Joint().count]
        print('Multiple Object Tracking (MOT) metrics:')
        printTable(metrics, motHeader=True)
    return (apAll, preAll, recAll), metrics
