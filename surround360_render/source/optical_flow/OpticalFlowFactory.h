/**
* Copyright (c) 2016-present, Facebook, Inc.
* All rights reserved.
*
* This source code is licensed under the BSD-style license found in the
* LICENSE_render file in the root directory of this subproject. An additional grant
* of patent rights can be found in the PATENTS file in the same directory.
*/

#pragma once

#include <string>

#include "PixFlow.h"
#include "VrCamException.h"

namespace surround360 {
namespace optical_flow {

using namespace std;
using namespace cv;

template <
  int MaxPercentage = 0 // how far to look when initializing flow
>
static OpticalFlowInterface* makeOpticalFlowByPercentage() {
  static const float kPyrScaleFactor                  = 0.9f;
  static const float kSmoothnessCoef                  = 0.001f;
  static const float kVerticalRegularizationCoef      = 0.01f;
  static const float kHorizontalRegularizationCoef    = 0.01f;
  static const float kGradientStepSize                = 0.5f;
  static const float kDownscaleFactor                 = 0.5f;
  static const float kDirectionalRegularizationCoef   = 0.0f;
  return new PixFlow<false, MaxPercentage>(
    kPyrScaleFactor,
    kSmoothnessCoef,
    kVerticalRegularizationCoef,
    kHorizontalRegularizationCoef,
    kGradientStepSize,
    kDownscaleFactor,
    kDirectionalRegularizationCoef
  );
}



static OpticalFlowInterface* makeOpticalFlowByName(const string flowAlgName) {

  if (flowAlgName == "pixflow_low") {
    return makeOpticalFlowByPercentage<0>();
  }

  if (flowAlgName == "pixflow_search_10") {
    return makeOpticalFlowByPercentage<10>();
  }

  if (flowAlgName == "pixflow_search_20") {
    return makeOpticalFlowByPercentage<20>();
  }

  if (flowAlgName == "pixflow_search_40") {
    return makeOpticalFlowByPercentage<40>();
  }

  throw VrCamException("unrecognized flow algorithm name: " + flowAlgName);
}

} // namespace optical_flow
} // namespace surround360
