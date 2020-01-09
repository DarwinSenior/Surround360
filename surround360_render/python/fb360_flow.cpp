#include <pybind11/pybind11.h>
#include "optical_flow/OpticalFlowFactory.h"
#include "optical_flow/OpticalFlowVisualization.h"
#include "optical_flow/NovelView.h"
#include "VrCamException.h"
#include "ndarray.hpp"
namespace py = pybind11;
using namespace surround360::optical_flow;
using namespace cv;

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

Mat visualize(const Mat& flow, const string& mode) {
  if (mode == "color") {
    return visualizeFlowColorWheel(flow);
  } else if ( mode == "gray" ) {
    return visualizeFlowAsGreyDisparity(flow);
  } else {
    throw surround360::VrCamException(
      "Unrecognized visualization mode: "
      + mode
      + ". Use either 'color' or 'gray'");
  }
}

class NVGen {
  public:
    NVGen(const Mat& imgL, const Mat& imgR): flow("pixflow_search_20") {
        Mat img1(imgL), img2(imgR);
        if (img1.channels() == 3) { cvtColor(img1, img1, COLOR_BGR2BGRA); }
        if (img2.channels() == 3) { cvtColor(img2, img2, COLOR_BGR2BGRA); }
      Mat mat;
      flow.prepare(img1, img2, mat, mat, mat, mat);
    }
    ~NVGen() {}
    Mat view(float t) {
      Mat left, right, merged;
      flow.generateNovelView(t, merged, left, right);
      return merged;
    }
  private:
    NovelViewGeneratorAsymmetricFlow flow;
};

PYBIND11_MODULE(fb360_flow, m) {
  m.doc() = "Optical Flow implemented in Facebook360";
  py::enum_<OpticalFlowInterface::DirectionHint>(m, "hint", py::arithmetic())
    .value("unknown", OpticalFlowInterface::DirectionHint::UNKNOWN)
    .value("right", OpticalFlowInterface::DirectionHint::RIGHT)
    .value("left", OpticalFlowInterface::DirectionHint::LEFT)
    .value("down", OpticalFlowInterface::DirectionHint::DOWN)
    .value("up", OpticalFlowInterface::DirectionHint::UP)
    .export_values();

  py::class_<NVGen>(m, "ViewGenerator")
    .def(py::init([](const Mat& left, const Mat& right){
          return std::unique_ptr<NVGen>(new NVGen(left, right));
    }))
    .def("view", &NVGen::view, "view at t", py::arg("t"));

  m.def("compute_flow", &compute_flow,
      "create an optical flow algorithm",
      py::arg("imgL"), py::arg("imgR"),
      py::arg("direction") = OpticalFlowInterface::DirectionHint::UNKNOWN,
      py::arg("method") = "pixflow_low");

  m.def("flow2vis", &visualize,
      "convert 2d flow to 1d grayscale disparity(gray) or 3d colorwheel(color)",
      py::arg("flow"), py::arg("mode") = "gray");
  m.def("warp", &NovelViewUtil::generateNovelViewSimpleCvRemap,
      "given an image and a flow vector field of the same size"
      "generate a new image by applying the flow to the input scaled by t",
      py::arg("srcImg"), py::arg("flow"), py::arg("t") = 1);
  m.def("consistency_check", &NovelViewUtil::consistencyCheck,
      R"(given left to right flow and right to left flow, create a consistency map for l2r flow)",
      py::arg("flowLtoR"), py::arg("flowRtoL"), py::arg("tolerance")=1.0);
}
