/*
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * @author: Rohit Girdhar (rgirdhar)
 */

#include "affine_channel_nd_op.h"
#include "caffe2/core/context_gpu.h"

namespace caffe2 {

namespace {
template <typename T>
__global__ void ScaleBiasForward(
    const int n,
    const T* in,
    const T* scale,
    const T* bias,
    const int scale_dim,
    const int hxw_dim,
    T* out) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int scale_index = (index / hxw_dim) % scale_dim;
    out[index] = in[index] * scale[scale_index] + bias[scale_index];
  }
}

template <typename T>
__global__ void ScaleForward(
    const int n,
    const T* in,
    const T* scale,
    const int scale_dim,
    const int hxw_dim,
    T* out) {
  CUDA_1D_KERNEL_LOOP(index, n) {
    const int scale_index = (index / hxw_dim) % scale_dim;
    out[index] = in[index] * scale[scale_index];
  }
}
} // namespace

template <>
bool AffineChannelNdOp<float, CUDAContext>::RunOnDevice() {
  auto& X = Input(0);
  auto& scale = Input(1);
  auto& bias = Input(2);
  auto* Y = Output(0);

  Y->ResizeLike(X);
  const int output_size = Y->size();
  int blob_size = 1;
  for (int i = 2; i < X.ndim(); i++) {
    blob_size *= X.dim32(i);
  }
  CAFFE_ENFORCE_EQ(X.dim32(1), scale.size());
  CAFFE_ENFORCE_EQ(X.dim32(1), bias.size());
  ScaleBiasForward<float><<<CAFFE_GET_BLOCKS(output_size),
                            CAFFE_CUDA_NUM_THREADS,
                            0, context_.cuda_stream()>>>(
      output_size, X.data<float>(), scale.data<float>(), bias.data<float>(),
      X.dim32(1), blob_size, Y->mutable_data<float>());
  return true;
}


template <>
bool AffineChannelNdGradientOp<float, CUDAContext>::RunOnDevice() {
  auto& scale = Input(0);
  auto& dY = Input(1);
  auto* dX = Output(0);

  // TODO(rbg): Implement gradients for scale and bias
  dX->ResizeLike(dY);
  int blob_size = 1;
  for (int i = 2; i < dY.ndim(); i++) {
    blob_size *= dY.dim32(i);
  }
  CAFFE_ENFORCE_EQ(dY.dim32(1), scale.size());
  ScaleForward<float><<<CAFFE_GET_BLOCKS(dX->size()),
                        CAFFE_CUDA_NUM_THREADS,
                        0, context_.cuda_stream()>>>(
      dY.size(), dY.data<float>(), scale.data<float>(),
      dY.dim32(1), blob_size, dX->mutable_data<float>());
  return true;
}


REGISTER_CUDA_OPERATOR(AffineChannelNd,
                       AffineChannelNdOp<float, CUDAContext>);
REGISTER_CUDA_OPERATOR(AffineChannelNdGradient,
                       AffineChannelNdGradientOp<float, CUDAContext>);
} // namespace caffe2
