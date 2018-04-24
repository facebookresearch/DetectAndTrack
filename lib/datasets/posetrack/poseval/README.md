This evaluation code base is a fork of an initial version of the
PoseTrack evaluation codebase released
[here](https://github.com/leonid-pishchulin/poseval/).

# Evaluation of Multi-Person Pose Estimation and Tracking

Created by Leonid Pishchulin

## Introduction

This README provides instructions how to evaluate your method's predictions on [PoseTrack Dataset](https://posetrack.net) locally or using evaluation server.

## Prerequisites

- numpy>=1.12.1
- pandas>=0.19.2
- scipy>=0.19.0

## Install
```
$ git clone https://github.com/leonid-pishchulin/poseval.git --recursive
$ cd poseval/py && export PYTHONPATH=$PWD/../py-motmetrics:$PYTHONPATH
```
## Data preparation

Evaluation requires ground truth (GT) annotations available at [PoseTrack](https://posetrack.net) and  your method's predictions. Both GT annotations and your predictions must be saved in json format. Following GT annotations, predictions must be stored per sequence using the same structure as GT annotations, and have the same filename as GT annotations. Example of json prediction structure:
```
{
   "annolist": [
       {
	   "image": [
	       {
		  "name": "images\/bonn_5sec\/000342_mpii\/00000001.jpg"
	       }
           ],
           "annorect": [
	       {
	           "x1": [625],
		   "y1": [94],
		   "x2": [681],
		   "y2": [178],
		   "score": [0.9],
		   "track_id": [0],
		   "annopoints": [
		       {
			   "point": [
			       {
			           "id": [0],
				   "x": [394],
				   "y": [173],
			       },
			       { ... }
			   ]
		       }
		   ]
		},
		{ ... }
	   ],
       },
       { ... }
   ]
}
```
Note: values of `track_id` must integers from the interval [0, 999].

We provide a possibility to convert a Matlab structure into json format.
```
$ cd poseval/matlab
$ matlab -nodisplay -nodesktop -r "mat2json('/path/to/dir/with/mat/files/'); quit"
```

## Metrics

This code allows to perform evaluation of per-frame multi-person pose estimation and evaluation of video-based multi-person pose tracking.

### Per-frame multi-person pose estimation

Average Precision (AP) metric is used for evaluation of per-frame multi-person pose estimation. Our implementation follows the measure proposed in [1] and requires predicted body poses with body joint detection scores as input. First, multiple body pose predictions are greedily assigned to the ground truth (GT) based on the highest PCKh [3]. Only single pose can be assigned to GT. Unassigned predictions are counted as false positives. Finally, part detection score is used to compute AP for each body part. Mean AP over all body parts is reported as well.

### Video-based pose tracking

Multiple Object Tracking (MOT) metrics [2] are used for evaluation of video-based pose tracking. Our implementation builds on the MOT evaluation code [4] and requires predicted body poses with tracklet IDs as input. First, for each frame, for each body joint class, distances between predicted locations and GT locations are computed. Then, predicted tracklet IDs and GT tracklet IDs are taken into account and all (prediction, GT) pairs with distances not exceeding PCKh [3] threshold are considered during global matching of predicted tracklets to GT tracklets for each particular body joint. Global matching minimizes the total assignment distance. Finally, Multiple Object Tracker Accuracy (MOTA), Multiple Object Tracker Precision (MOTP), Precision, and Recall metrics are computed. We report MOTA metric for each body joint class and average over all body joints, while for MOTP, Precision, and Recall we report averages only.

## Evaluation (local)

Evaluation code has been tested in Linux and Ubuntu OS. Evaluation takes as input path to directory with GT annotations and path to directory with predictions. See "Data preparation" for details on prediction format.

```
$ git clone https://github.com/leonid-pishchulin/poseval.git --recursive
$ cd poseval/py && export PYTHONPATH=$PWD/../py-motmetrics:$PYTHONPATH
$ python evaluate.py \
  --groundTruth=/path/to/annotations/val/ \
  --predictions=/path/to/predictions \
  --evalPoseTracking \
  --evalPoseEstimation
```

Evaluation of multi-person pose estimation requires joint detection scores, while evaluation of pose tracking requires predicted tracklet IDs per pose.

## Evaluation (server)

In order to evaluate using evaluation server, zip your directory containing json prediction files and submit at https://posetrack.net. Shortly you will receive an email containing evaluation results. **Prior to submitting your results to evaluation server, make sure you are able to evaluate locally on val set to avoid issues due to incorrect formatting of predictions.**

## References

[1] DeepCut: Joint Subset Partition and Labeling for Multi Person Pose Estimation. L. Pishchulin, E. Insafutdinov, S. Tang, B. Andres, M. Andriluka, P. Gehler, and B. Schiele. In CVPR'16

[2] Evaluating multiple object tracking performance: the CLEAR MOT metrics. K. Bernardin and R. Stiefelhagen. EURASIP J. Image Vide.'08

[3] 2D Human Pose Estimation: New Benchmark and State of the Art Analysis. M. Andriluka, L. Pishchulin, P. Gehler, and B. Schiele. In CVPR'14

[4] https://github.com/cheind/py-motmetrics

For further questions and details, contact PoseTrack Team <mailto:admin@posetrack.net>
