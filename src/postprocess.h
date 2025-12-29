#pragma once
#include <vector>

struct Det {
  float x1,y1,x2,y2;
  float conf;
  int cls;
};

// output0: float array with shape [1,8,8400] (channels-first)
// roi: ROI size
std::vector<Det> decode_yolo_1x8xN_f32(
  const float* out, int roi,
  float conf_thres, float iou_thres,
  int ox, int oy, int full_w, int full_h,
  float* out_max_score
);
