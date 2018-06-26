# Use Caffe2 image as parent image
FROM caffe2/caffe2:snapshot-py2-cuda9.0-cudnn7-ubuntu16.04

RUN mv /usr/local/caffe2 /usr/local/caffe2_build
ENV Caffe2_DIR /usr/local/caffe2_build

ENV PYTHONPATH /usr/local/caffe2_build:${PYTHONPATH}
ENV LD_LIBRARY_PATH /usr/local/caffe2_build/lib:${LD_LIBRARY_PATH}

RUN apt-get -y update

# Clone the Detectron repository
RUN git clone https://github.com/facebookresearch/DetectAndTrack.git /detectandtrack

RUN apt-get install -y libeigen3-dev python-tk

# Install Python dependencies
RUN cd /detectandtrack \
        && pip install -r requirements.txt

# Install the COCO API
RUN git clone https://github.com/cocodataset/cocoapi.git /cocoapi
WORKDIR /cocoapi/PythonAPI
RUN make install

# Go to Densepose root
WORKDIR /detectandtrack/lib

# Set up Python modules
RUN make

# [Optional] Build custom ops
RUN make ops

# Go to Densepose root
WORKDIR /detectandtrack

RUN apt-get clean

