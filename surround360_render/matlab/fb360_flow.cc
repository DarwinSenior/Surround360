#include "MxArray.hpp"
#include "optical_flow/NovelView.h"
#include "optical_flow/OpticalFlowFactory.h"

using namespace cv;
using namespace std;
using namespace surround360::optical_flow;

Mat compute_flow(const Mat& imgL, const Mat& imgR,
    OpticalFlowInterface::DirectionHint direction, const string& method) {
  Mat flow, prevFlow, prevImgL, prevImgR;

  Mat img1, img2;
  if (imgL.channels() == 3) {
    cvtColor(imgL, img1, COLOR_BGR2BGRA);
  } else {
    img1 = imgL.clone();
  }
  if (imgR.channels() == 3) {
    cvtColor(imgR, img2, COLOR_BGR2BGRA);
  } else {
    img2 = imgR.clone();
  }
  if (imgL.depth() == CV_32F) { img1 = img1 * 255; }
  if (imgR.depth() == CV_32F) { img2 = img2 * 255; }


  auto ref = makeOpticalFlowByName(method);
  ref->computeOpticalFlow(img1, img2,
      prevFlow, prevImgL, prevImgR, flow, direction);
  delete ref;
  return flow;
}

/**
 * input function construct
 */
void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray* prhs[]) {
  if (nrhs < 3) mexErrMsgTxt("FB360_flow usage: imgL, imgR, method");
  auto imgL = MxArray(prhs[0]).toMat();
  auto imgR = MxArray(prhs[1]).toMat();
  auto method = MxArray(prhs[2]).toString();
  auto flow = compute_flow(imgL, imgR, OpticalFlowInterface::DirectionHint::UNKNOWN, method);
  plhs[0] = MxArray(flow);
}
