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

namespace caffe2 {

REGISTER_CPU_OPERATOR(AffineChannelNd,
                      AffineChannelNdOp<float, CPUContext>);
REGISTER_CPU_OPERATOR(AffineChannelNdGradient,
                      AffineChannelNdGradientOp<float, CPUContext>);

// Input: X, scale, bias; Output: Y
OPERATOR_SCHEMA(AffineChannelNd).NumInputs(3).NumOutputs(1).AllowInplace(
  {{0, 0}});
// Input: X, scale, dY; Output: dX
OPERATOR_SCHEMA(AffineChannelNdGradient).NumInputs(2).NumOutputs(
  1).AllowInplace({{1, 0}});

// TODO(rbg): Implement gradients for scale and bias
class GetAffineChannelNdGradient : public GradientMakerBase {
  using GradientMakerBase::GradientMakerBase;
  vector<OperatorDef> GetGradientDefs() override {
    return SingleGradientDef(
        "AffineChannelNdGradient", "",
        vector<string>{I(1), GO(0)},
        vector<string>{GI(0)});
  }
};

REGISTER_GRADIENT(AffineChannelNd, GetAffineChannelNdGradient);

} // namespace caffe2
