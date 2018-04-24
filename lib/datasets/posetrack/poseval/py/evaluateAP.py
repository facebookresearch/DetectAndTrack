import numpy as np
import json
import os
import sys

import eval_helpers


def computeMetrics(scoresAll, labelsAll, nGTall):
    apAll = np.zeros((nGTall.shape[0] + 1, 1))
    recAll = np.zeros((nGTall.shape[0] + 1, 1))
    preAll = np.zeros((nGTall.shape[0] + 1, 1))
    # iterate over joints
    for j in range(nGTall.shape[0]):
        scores = np.zeros([0, 0], dtype=np.float32)
        labels = np.zeros([0, 0], dtype=np.int8)
        # iterate over images
        for imgidx in range(nGTall.shape[1]):
            scores = np.append(scores, scoresAll[j][imgidx])
            labels = np.append(labels, labelsAll[j][imgidx])
        # compute recall/precision values
        nGT = sum(nGTall[j, :])
        precision, recall, scoresSortedIdxs = eval_helpers.computeRPC(scores, labels, nGT)
        if (len(precision) > 0):
            apAll[j] = eval_helpers.VOCap(recall, precision) * 100
            preAll[j] = precision[len(precision) - 1] * 100
            recAll[j] = recall[len(recall) - 1] * 100
    apAll[nGTall.shape[0]] = apAll[:nGTall.shape[0], 0].mean()
    recAll[nGTall.shape[0]] = recAll[:nGTall.shape[0], 0].mean()
    preAll[nGTall.shape[0]] = preAll[:nGTall.shape[0], 0].mean()

    return apAll, preAll, recAll


def evaluateAP(gtFramesAll, prFramesAll):

    distThresh = 0.5

    # assign predicted poses to GT poses
    scoresAll, labelsAll, nGTall, _ = eval_helpers.assignGTmulti(gtFramesAll, prFramesAll, distThresh)

    # compute average precision (AP), precision and recall per part
    apAll, preAll, recAll = computeMetrics(scoresAll, labelsAll, nGTall)

    return apAll, preAll, recAll

