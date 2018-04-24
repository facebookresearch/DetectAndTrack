import numpy as np
import json
import os
import sys

import eval_helpers

def computeDist(gtFrames,prFrames):
    assert(len(gtFrames) == len(prFrames))

    nJoints = eval_helpers.Joint().count
    distAll = {}
    for pidx in range(nJoints):
        distAll[pidx] = np.zeros([0,0])
        
    for imgidx in range(len(gtFrames)):
        # ground truth
        gtFrame = gtFrames[imgidx]
        # prediction
        detFrame = prFrames[imgidx]
        if (gtFrames[imgidx]["annorect"] != None):
            for ridx in range(len(gtFrames[imgidx]["annorect"])):
                rectGT = gtFrames[imgidx]["annorect"][ridx]
                rectPr = prFrames[imgidx]["annorect"][ridx]
                if ("annopoints" in rectGT.keys() and rectGT["annopoints"] != None):
                    pointsGT = rectGT["annopoints"][0]["point"]
                    pointsPr = rectPr["annopoints"][0]["point"]
                    for pidx in range(len(pointsGT)):
                        pointGT = [pointsGT[pidx]["x"][0],pointsGT[pidx]["y"][0]]
                        idxGT = pointsGT[pidx]["id"][0]
                        p = eval_helpers.getPointGTbyID(pointsPr,idxGT)
                        if (len(p) > 0 and
                            (type(p["x"][0]) == int or type(p["x"][0]) == float) and
                            (type(p["y"][0]) == int or type(p["y"][0]) == float)):
                            pointPr = [p["x"][0],p["y"][0]]
                            # compute distance between GT and prediction
                            d = np.linalg.norm(np.subtract(pointGT,pointPr))
                            # compute head size for distance normalization
                            headSize = eval_helpers.getHeadSize(rectGT["x1"][0],rectGT["y1"][0],
                                                                rectGT["x2"][0],rectGT["y2"][0])
                            # normalize distance
                            dNorm = d/headSize
                        else:
                            dNorm = np.inf
                        distAll[idxGT] = np.append(distAll[idxGT],[[dNorm]])

    return distAll


def computePCK(distAll,distThresh):

    pckAll = np.zeros([len(distAll)+1,1])
    nCorrect = 0
    nTotal = 0
    for pidx in range(len(distAll)):
        idxs = np.argwhere(distAll[pidx] <= distThresh)
        pck = 100.0*len(idxs)/len(distAll[pidx])
        pckAll[pidx,0] = pck
        nCorrect += len(idxs)
        nTotal   += len(distAll[pidx])
    pckAll[len(distAll),0] = 100.0*nCorrect/nTotal
    
    return pckAll


def evaluatePCKh(gtFramesAll,prFramesAll):

    distThresh = 0.5
    
    # compute distances
    distAll = computeDist(gtFramesAll,prFramesAll)

    # compute PCK metric
    pckAll = computePCK(distAll,distThresh)

    return pckAll
