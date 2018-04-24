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

#ifndef AFFINE_CHANNEL_ND_OP_H_
#define AFFINE_CHANNEL_ND_OP_H_

#include "caffe2/core/context.h"
#include "caffe2/core/logging.h"
#include "caffe2/core/operator.h"
#include "caffe2/utils/math.h"

namespace caffe2 {

template <typename T, class Context>
class AffineChannelNdOp final : public Operator<Context> {
 public:
  AffineChannelNdOp(const OperatorDef& operator_def, Workspace* ws)
      : Operator<Context>(operator_def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // Lazy, not implementing the CPU version
    CAFFE_NOT_IMPLEMENTED;
    return true;
  }
};

template <typename T, class Context>
class AffineChannelNdGradientOp final : public Operator<Context> {
 public:
  AffineChannelNdGradientOp(const OperatorDef& def, Workspace* ws)
      : Operator<Context>(def, ws) {}
  USE_OPERATOR_CONTEXT_FUNCTIONS;

  bool RunOnDevice() override {
    // Lazy, not implementing the CPU version
    CAFFE_NOT_IMPLEMENTED;
    return true;
  }
};

} // namespace caffe2

#endif // AFFINE_CHANNEL_ND_OP_H_
