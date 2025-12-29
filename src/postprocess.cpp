#include "postprocess.h"
#include <algorithm>
#include <cmath>

static inline float clampf(float v, float lo, float hi) {
  return v < lo ? lo : (v > hi ? hi : v);
}

static float iou(const Det& a, const Det& b)
{
  float xx1 = std::max(a.x1, b.x1);
  float yy1 = std::max(a.y1, b.y1);
  float xx2 = std::min(a.x2, b.x2);
  float yy2 = std::min(a.y2, b.y2);
  float w = std::max(0.0f, xx2 - xx1);
  float h = std::max(0.0f, yy2 - yy1);
  float inter = w * h;
  float areaA = std::max(0.0f, a.x2-a.x1) * std::max(0.0f, a.y2-a.y1);
  float areaB = std::max(0.0f, b.x2-b.x1) * std::max(0.0f, b.y2-b.y1);
  float uni = areaA + areaB - inter + 1e-6f;
  return inter / uni;
}

static std::vector<Det> nms(std::vector<Det>& dets, float iou_thres)
{
  std::sort(dets.begin(), dets.end(), [](const Det& a, const Det& b){ return a.conf > b.conf; });
  std::vector<Det> keep;
  std::vector<char> rm(dets.size(), 0);

  for (size_t i=0;i<dets.size();i++){
    if (rm[i]) continue;
    keep.push_back(dets[i]);
    for (size_t j=i+1;j<dets.size();j++){
      if (rm[j]) continue;
      if (dets[i].cls != dets[j].cls) continue;
      if (iou(dets[i], dets[j]) > iou_thres) rm[j]=1;
    }
  }
  return keep;
}

std::vector<Det> decode_yolo_1x8xN_f32(
  const float* out, int roi,
  float conf_thres, float iou_thres,
  int ox, int oy, int full_w, int full_h,
  float* out_max_score)
{
  const int N = 3549;
  float max_score = 0.0f;

  std::vector<Det> dets;
  dets.reserve(256);

  for (int i = 0; i < N; i++) {
    float cx = out[0*N + i];
    float cy = out[1*N + i];
    float w  = out[2*N + i];
    float h  = out[3*N + i];

    int best_cls = 0;
    float best_sc = out[4*N + i];
    for (int c = 1; c < 4; c++) {
      float sc = out[(4+c)*N + i];
      if (sc > best_sc) { best_sc = sc; best_cls = c; }
    }

    if (best_sc > max_score) max_score = best_sc;
    if (best_sc < conf_thres) continue;

    // If normalized, scale up
    if (cx <= 2.0f && cy <= 2.0f && w <= 2.0f && h <= 2.0f) {
      cx *= roi; cy *= roi; w *= roi; h *= roi;
    }

    Det d;
    d.x1 = cx - w * 0.5f + ox;
    d.y1 = cy - h * 0.5f + oy;
    d.x2 = cx + w * 0.5f + ox;
    d.y2 = cy + h * 0.5f + oy;
    d.conf = best_sc;
    d.cls = best_cls;

    d.x1 = clampf(d.x1, 0, (float)full_w-1);
    d.y1 = clampf(d.y1, 0, (float)full_h-1);
    d.x2 = clampf(d.x2, 0, (float)full_w-1);
    d.y2 = clampf(d.y2, 0, (float)full_h-1);

    dets.push_back(d);
  }

  if (out_max_score) *out_max_score = max_score;
  if (dets.empty()) return dets;
  return nms(dets, iou_thres);
}
