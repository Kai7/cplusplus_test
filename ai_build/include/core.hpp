#ifndef _AI_CORE_HPP_
#define _AI_CORE_HPP_

#define AI_EXPORT __attribute__((visibility("default")))

#include "cv/core.hpp"

class AI_EXPORT AIMat2D {
public:
  AIMat2D();
  AIMat2D(uint32_t row, uint32_t col);
  AIMat2D(uint32_t row, uint32_t col, float **data);
  ~AIMat2D();

  float Trace();

private:
  cv::Mat2D _m;
};

#endif