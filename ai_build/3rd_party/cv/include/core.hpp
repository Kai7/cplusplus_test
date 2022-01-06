#ifndef _CV_CORE_HPP_
#define _CV_CORE_HPP_

#include "stdint.h"
#include <vector>

namespace cv {

class Mat2D {
public:
  Mat2D();
  Mat2D(uint32_t row, uint32_t col);
  Mat2D(uint32_t row, uint32_t col, float **data);
  ~Mat2D();

  uint32_t Row();
  uint32_t Col();
  float at(uint32_t i, uint32_t j);

#if 1
  float Sum();
  float RowSum(uint32_t i);
  float ColSum(uint32_t j);
  uint32_t Size();
#endif

private:
  uint32_t _row;
  uint32_t _col;
  float **_data;
};

#if 0
uint32_t getSize_0(Mat2D m);
uint32_t getSize_1(Mat2D m);
uint32_t getSize_2(Mat2D m);

float add_0(float a, float b);
float add_1(float a, float b);
float add_2(float a, float b);
#endif

} // namespace cv

#endif