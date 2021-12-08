#ifndef _CV_CORE_HPP_
#define _CV_CORE_HPP_

#include "stdint.h"
#include <vector>

class Mat2D{
public:
  Mat2D();
  Mat2D(uint32_t row, uint32_t col);
  Mat2D(uint32_t row, uint32_t col, float **data);
  ~Mat2D();

  uint32_t Row();
  uint32_t Col();
  float at(uint32_t i, uint32_t j);

  float Sum();

private:
  uint32_t _row;
  uint32_t _col;
  float **_data;
};

#endif