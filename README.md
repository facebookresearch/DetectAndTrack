# Detect And Track: Efficient Pose Estimation in Videos

![Eg1](https://rohitgirdhar.github.io/DetectAndTrack/assets/posetrack1.gif)
![Eg2](https://rohitgirdhar.github.io/DetectAndTrack/assets/posetrack2.gif)

<p><img src="https://rohitgirdhar.github.io/DetectAndTrack/assets/cup.png" width="50px" align="center" /> Ranked <b>first</b> in the keypoint tracking task of the <a href="https://posetrack.net/leaderboard.php">ICCV 2017 PoseTrack challenge</a>! (entry: ProTracker)</p>


[[project page](https://rohitgirdhar.github.io/DetectAndTrack/)] [[paper](https://arxiv.org/abs/1712.09184)]

If this code helps with your work, please cite:

R. Girdhar, G. Gkioxari, L. Torresani, M. Paluri and D. Tran. **Detect-and-Track: Efficient Pose Estimation in Videos.** IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2018.

```
@inproceedings{girdhar2018detecttrack,
    title = {{Detect-and-Track: Efficient Pose Estimation in Videos}},
    author = {Girdhar, Rohit and Gkioxari, Georgia and Torresani, Lorenzo and Paluri, Manohar and Tran, Du},
    booktitle = {CVPR},
    year = 2018
}
```

## Requirements

This code was developed and tested on NVIDIA P100 (16GB), M40 (12GB) and 1080Ti (11GB) GPUs. Training requires at least 4 GPUs for most configurations, and some were trained with 8 GPUs. It might be possible to train on a single GPU by scaling down the learning rate and scaling up the iteration schedule, but we have not tested all possible setups. Testing can be done on a single GPU. Unfortunately it is currently not possible to run this on a CPU as some ops do not have CPU implementations.

## Installation

If you have used [Detectron](https://github.com/facebookresearch/Detectron), you should have most of the prerequisites installed, except some required for [PoseTrack evaluation](https://github.com/leonid-pishchulin/poseval/).
In any case, the following instructions should get you started. I would strongly recommend using anaconda, as it makes it really easy to install most libraries required to compile caffe2 and other ops. First start by cloning this code:

```bash
$ git clone https://github.com/facebookresearch/DetectAndTrack.git
$ cd DetectAndTrack
```

### Pre-requisites and software setup

The code was tested with the following setup:

0. CentOS 6.5
1. Anaconda (python 2.7)
2. OpenCV 3.4.1
3. GCC 4.9
4. CUDA 9.0
5. cuDNN 7.1.2
6. numpy 1.14.2 (needs >=1.12.1, for the [poseval](https://github.com/leonid-pishchulin/poseval/) evaluation scripts)

The [`all_pkg_versions.txt`](all_pkg_versions.txt) file contains the exact versions of packages that should work with this code. To avoid conflicting packages, I would suggest creating a new environment in conda, and installing all the requirements in there. It can be done by:

```bash
$ export ENV_NAME="detect_and_track"  # or any other name you prefer
$ conda create --name $ENV_NAME --file all_pkg_versions.txt python=2.7 anaconda
$ source activate $ENV_NAME
```

If you are using an old OS (like CentOS 6.5), you might want to install versions of packages compatible with the GLIBC library on your system. On my system with GLIBC 2.12, using libraries from the `conda-forge` channel seemed to work fine. To use it, simply change the `conda create` command by adding a `-c conda-forge`.

### Install Caffe2

Follow the instructions from the [caffe2 installation instructions](https://caffe2.ai/docs/getting-started.html?platform=ubuntu&configuration=compile). I describe what worked for me on CentOS 6.5 next. The code was tested with [b4e158](https://github.com/caffe2/caffe2/tree/b4e1588130198b6e98e4d0acf5b340015473e562) commit release of C2.

```bash
$ cd ..
$ git clone --recursive https://github.com/caffe2/caffe2.git && cd caffe2
$ git submodule update --init
$ mkdir build && cd build
$ export CONDA_PATH=/path/to/anaconda2  # Set this path as per your anaconda installation
$ export CONDA_ENV_PATH=$CONDA_PATH/envs/$ENV_NAME
$ cmake \
	-DCMAKE_PREFIX_PATH=$CONDA_ENV_PATH \
	-DCMAKE_INSTALL_PREFIX=$CONDA_ENV_PATH \
	-Dpybind11_INCLUDE_DIR=$CONDA_ENV_PATH/include \
	-DCMAKE_THREAD_LIBS_INIT=$CONDA_ENV_PATH/lib ..
$ make -j32
$ make install -j32  # This installs into the environment
```

This should install caffe2 on your anaconda. Please refer to the official caffe2 installation instructions for more information and help.


### Compile some custom ops

We need one additional op for running the 3D models, and is provided as `lib/ops/affine_channel_nd_op.*`. It can be installed following instructions from [here](https://github.com/facebookresearch/Detectron/blob/master/INSTALL.md#advanced-topic-custom-operators-for-new-research-projects), or:

```bash
$ cd ../DetectAndTrack/lib
$ make && make ops
$ cd ..
$ python tests/test_zero_even_op.py  # test that compilation worked
```

In case this does not work, an alternative is to copy over the `lib/ops/affine_channel_nd_op.*` files into the caffe2 detectron module folder (`caffe2/modules/detectron/`), and recompiling caffe2. This would also make this additional op available to caffe2.


### Install the COCO API

Since the datasets are represented using COCO conventions in Detectron code base, we need the COCO API to be able to read the train/test files. It can be installed by:

```bash
$ # COCOAPI=/path/to/clone/cocoapi
$ git clone https://github.com/cocodataset/cocoapi.git $COCOAPI
$ cd $COCOAPI/PythonAPI
$ # Install into global site-packages
$ make install
$ # Alternatively, if you do not have permissions or prefer
$ # not to install the COCO API into global site-packages
$ python2 setup.py install --user
```

## Dataset and Evaluation

We use a slightly modified version of the PoseTrack dataset where we rename the frames to follow `%08d` format, with first frame indexed as 1 (i.e. `00000001.jpg`).
Download and extract the data from [PoseTrack download page](https://posetrack.net/users/download.php) into `lib/datasets/data/PoseTrack` (or create a symlink to this location).
Then, rename the frames for each video to be named as described above, or use [`tools/gen_posetrack_json.py`](tools/gen_posetrack_json.py), which converts the data and generates labels in the JSON format compatible with Detectron.
We already provide the corresponding training/validation/testing JSON files in [`lib/datasets/lists/PoseTrack/v1.0`](lib/datasets/lists/PoseTrack/v1.0), which have already been converted to the COCO format. The paths to the data are hardcoded in `lib/datasets/json_dataset.py` file.

For evaluation, the code includes a modified version of [poseval](https://github.com/leonid-pishchulin/poseval) with multi-processing for faster results. We have verified that the number from this code matches what we get from the [evaluation server](https://posetrack.net/). Since evaluation is done using provided code, we also need the provided MAT/JSON files with labels which are used by this code to compute the final number. You can download these files from [here](https://cmu.box.com/shared/static/gakn52qoxzfzrvmo4c0gfxt4ja1i4g0b.tar), and extract them as `lib/datasets/data/PoseTrackV1.0_Annots_val_json`.

**NOTE**: Extract the val files into a fast local disk. For some reason, I am seeing slightly different performance if these files are stored on a NFS directory. This might be an issue with the evaluation code (the organizers also found slightly different numbers using their code locally and on the evaluation server), but since the difference is pretty marginal (~0.1% overall MOTA), I am ignoring it for now. When storing the val files on a fast local disk, I can exactly reproduce the performance reported in the paper. However on any disk, the trends should remain the same, with only minor variations in the absolute numbers.

## Running the code

We provide a nifty little script `launch.py` that can take care of running any train/test/tracking workflows. Similar to Detectron, each experiment is completely defined by a YAML config file. We provide the config files required to reproduce our reported performance in the paper. In general, the script can be used as follows:

```bash
$ export CUDA_VISIBLE_DEVICES="0,1,2,3"  # set the subset of GPUs on the current node to use. Count must be same as NUM_GPUS set in the config
$ python launch.py --cfg/-c /path/to/config.yaml --mode/-m [train (default)/test/track/eval] ...[other config opts]...
```

The `/path/to/config.yaml` is the path to a YAML file with the experiment configuration (see `config` directory for some examples). `mode` defines whether you want to run training/testing/tracking/evaluation, and other config opts refer to *any* other config option (see `lib/core/config.py` for the full list). This command line config option has the highest precedence, so it will override any defaults or specifications in the YAML file, making it a quick way to experiment with specific configs. We show examples in the following sections.

Before starting, create an empty `outputs/` folder in the root directory. This can also be sym-linked to some large disk, as we will be storing all output models, files into this directory. The naming convention will be `outputs/path/to/config/file.yaml/`, and will contain `.pkl` model files, detection files etc. For ease of use, the training code will automatically run testing, which automatically runs tracking, which in turn automatically runs evaluation and produces the final performance.


### Running tracking and evaluating pre-trained, pre-tested models

We provide pre-trained models and files in a directory [here](https://cmu.box.com/s/qn5dnrv6pvrolcr16buxk28ctoizinu8). You can optionally download the whole directory as `pretrained_models/` in the root directory, or can download individual models you end up needing.

First, lets start by simply running tracking and evaluating the performance of our best models (that won the PoseTrack challenge). We will directly use the output detections on the val set and run tracking (we will get to testing and generating this detections later). As you will notice, the tracking is super fast, and obtains strong performance. Run the standard hungarian matching as follows:

```bash
$ python launch.py \
	--cfg configs/video/2d_best/01_R101_best_hungarian-4GPU.yaml \
	--mode track \
	TRACKING.DETECTIONS_FILE pretrained_models/configs/video/2d_best/01_R101_best_hungarian.yaml/detections.pkl
# Can also run with greedy matching
$ python launch.py \
	--cfg configs/video/2d_best/01_R101_best_hungarian-4GPU.yaml \
	--mode track \
	TRACKING.DETECTIONS_FILE pretrained_models/configs/video/2d_best/01_R101_best_hungarian.yaml/detections.pkl \
	TRACKING.BIPARTITE_MATCHING_ALGO greedy
```
It should produce the following performance, as also reported in the paper:

| Algorithm | mAP (overall) | MOTA (overall) |
|-----------|---------------|----------------|
| Hungarian | 60.6          | 55.2           |
| Greedy    | 60.6          | 55.1           |


#### Upper bound experiments

As another example, we can try to reproduce the upper bound performance:

```bash
# Perfect tracking
$ python launch.py \
	--cfg configs/video/2d_best/01_R101_best_hungarian-4GPU.yaml \
	--mode track \
	TRACKING.DETECTIONS_FILE pretrained_models/configs/video/2d_best/01_R101_best_hungarian.yaml/detections.pkl \
	TRACKING.DEBUG.UPPER_BOUND_4_EVAL_UPPER_BOUND True
# Perfect keypoints
$ python launch.py \
	--cfg configs/video/2d_best/01_R101_best_hungarian-4GPU.yaml \
	--mode track \
	TRACKING.DETECTIONS_FILE pretrained_models/configs/video/2d_best/01_R101_best_hungarian.yaml/detections.pkl \
	TRACKING.DEBUG.UPPER_BOUND_5_GT_KPS_ONLY True
# Perfect keypoints and tracks
$ python launch.py \
	--cfg configs/video/2d_best/01_R101_best_hungarian-4GPU.yaml \
	--mode track \
	TRACKING.DETECTIONS_FILE pretrained_models/configs/video/2d_best/01_R101_best_hungarian.yaml/detections.pkl \
	TRACKING.DEBUG.UPPER_BOUND_5_GT_KPS_ONLY True \
	TRACKING.DEBUG.UPPER_BOUND_4_EVAL_UPPER_BOUND True
```
This should obtain

| Setup |  mAP (overall) | MOTA (overall) |
|-----------|-------------|----------------|
| Perfect tracking |    60.6 | 57.6   |
| Perfect keypoints    |  82.9 |  78.4  |
| Perfect keypoints + tracking | 82.9  | 82.9 |


### Testing pre-trained models

We can also compute the detections file from a pre-trained model, like following. This will automatically also run tracking and evaluation, to produce the final number. Make sure to use NUM_GPUS is same as the GPUs you want to test on (as set in `CUDA_VISIBLE_DEVICES`).

```bash
$ python launch.py \
	--cfg configs/video/2d_best/01_R101_best_hungarian-4GPU.yaml \
	--mode test \
	TEST.WEIGHTS pretrained_models/configs/video/2d_best/01_R101_best_hungarian.yaml/model_final.pkl
```
This should reproduce the performance reported above, 55.2% MOTA and 60.6% mAP. Similarly, you can run testing for any model provided, using the corresponding config file.

### Training models

The models reported in the paper were originally trained on 8xP100 (16GB) GPUs. Since many users might not have access to such GPUs, we also provide alternative configurations that have been trained/tested on 1080Ti (11GB) GPUs. To reduce the memory requirement, we reduce the batch size, and scale down the learning rate by the same factor. We scale up the iteration schedule (total iterations, step size) by the same factor as well, and so far have obtained nearly similar performance.

We show some example runs next. We also provide the trained models and detections files in the `pretrained_models` folder, which can be used to reproduce the numbers. Training would also reproduce nearly same performance, except for some minor random variation.

#### 2D Mask R-CNN models

```bash
# Trained on 8xP100 GPUs
$ python launch.py -c configs/video/2d_best/01_R101_best_hungarian.yaml -m train
# Trained on 4x1080Ti GPUs
$ python launch.py -c configs/video/2d_best/01_R101_best_hungarian-4GPU.yaml -m train
```

| Config | mAP | MOTA |
|--------|-----|------|
| 8 GPU | 60.6 | 55.2 |
| 4 GPU | 61.3 | 55.9 |

So the 4 GPU configuration, though takes longer, obtains similar (in fact better, in this case) performance.

#### 3D Mask R-CNN models

The 3D Mask R-CNN models can be trained/tested the exact same way as well.

Trained on P100s (reported in paper, table 6)
```bash
# 2D model, pre-trained on ImNet
$ python launch.py -c configs/video/3d/02_R-18_PTFromImNet.yaml
# 3D model, pre-trained on ImNet
$ python launch.py -c configs/video/3d/04_R-18-3D_PTFromImNet.yaml
# 2D model, pre-trained on COCO
$ python launch.py -c configs/video/3d/01_R-18_PTFromCOCO.yaml
# 3D model, pre-trained on COCO
$ python launch.py -c configs/video/3d/03_R-18-3D_PTFromCOCO.yaml
```

| Model type | Pre-training | mAP | MOTA |
|------------|--------------|-----|------|
| 2D | ImageNet | 14.8 | 3.2 |
| 3D | ImageNet | 16.7 | 4.3 |
| 2D | COCO | 19.9 | 14.1 |
| 3D | COCO | 22.6 | 15.4 |

Trained on 1080Ti. Since the GPU memory is smaller, I can no longer run the 3D models with the same batch size as on P100 (2D models can still be run though). Here I am showing that performance is relatively stable even when running with lower batch size but training longer (to effectively get same number of epochs). All 4/8GPU configurations are equivalent: the learning rate was scaled down and number of steps/step size were scaled up.

```bash
# 2D model, pre-trained on ImNet
$ python launch.py -c configs/video/3d/02_R-18_PTFromImNet-4GPU.yaml
# 3D model, pre-trained on ImNet
$ python launch.py -c configs/video/3d/04_R-18-3D_PTFromImNet-8GPU-BATCH1.yam
# 3D model, pre-trained on ImNet
$ python launch.py -c configs/video/3d/04_R-18-3D_PTFromImNet-4GPU-BATCH1.yaml
# 2D model, pre-trained on COCO
$ python launch.py -c configs/video/3d/01_R-18_PTFromCOCO-4GPU.yaml
# 3D model, pre-trained on COCO (8 GPU)
$ python launch.py -c configs/video/3d/03_R-18-3D_PTFromCOCO-8GPU-BATCH1.yaml
# 3D model, pre-trained on COCO (4 GPU)
$ python launch.py -c configs/video/3d/03_R-18-3D_PTFromCOCO-4GPU-BATCH1.yaml
```

| Model type | #GPU | Pre-training | mAP | MOTA |
|------------|------|--------|-----|------|
| 2D | 4 | ImageNet | 13.9 | 2.3 |
| 3D | 8 | ImageNet | 16.6 | 4.1 |
| 3D | 4 | ImageNet | 16.7 | 5.3 |
| 2D | 4 | COCO | 19.5 | 13.8 |
| 3D | 8 | COCO | 22.9 | 16.0 |
| 3D | 4 | COCO | 22.5 | 15.6 |



## Known issues

Due to a bug in Caffe2, the multi-GPU training will normally work on machines where peer access within GPUs is enabled. There exists a workaround for machines where this is not the case. As mentioned in [this issue](https://github.com/facebookresearch/Detectron/issues/32), you can set 'USE_NCCL True' in config (or when running) to be able to run on other machines, though it is susceptible to deadlocks and hang-ups. For example, run as follows:

```bash
$ python launch.py -c /path/to/config.yaml -m train USE_NCCL True
```


## Acknowledgements

The code was built upon an initial version of the [Detectron](https://github.com/facebookresearch/Detectron) code base. Many thanks to the original authors for making their code available!


## License
DetectAndTrack is Apache 2.0 licensed, as found in the LICENSE file.
