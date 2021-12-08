#ifndef _AI_CORE_HPP_
#define _AI_CORE_HPP_

#include "cv/core.hpp"

class AIMat2D {
public:
  AIMat2D();
  AIMat2D(uint32_t row, uint32_t col);
  AIMat2D(uint32_t row, uint32_t col, float **data);
  ~AIMat2D();

  float Trace();

private:
  Mat2D _m;
};

#endif