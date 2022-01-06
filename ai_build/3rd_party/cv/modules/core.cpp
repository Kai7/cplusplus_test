#include "core.hpp"
#include "assert.h"
#include <cstring>

namespace cv {

Mat2D::Mat2D() : _row(0), _col(0), _data(nullptr) {}

Mat2D::Mat2D(uint32_t row, uint32_t col) : _row(row), _col(col) {
  _data = new float *[row];
  for (uint32_t i = 0; i < row; i++) {
    _data[i] = new float[col];
  }
}

Mat2D::Mat2D(uint32_t row, uint32_t col, float **data) : _row(row), _col(col) {
  _data = new float *[row];
  for (uint32_t i = 0; i < row; i++) {
    _data[i] = new float[col];
    memcpy(_data[i], data[i], sizeof(float) * col);
  }
}

Mat2D::~Mat2D() {
  for (uint32_t i = 0; i < _row; i++) {
    delete[] _data[i];
  }
  delete[] _data;
  _data = nullptr;
  _row = 0;
  _col = 0;
}

uint32_t Mat2D::Col() { return _col; }

uint32_t Mat2D::Row() { return _row; }

float Mat2D::at(uint32_t i, uint32_t j) {
  assert(i < _row && j < _col);
  return _data[i][j];
}

#if 1
float Mat2D::Sum() {
  float sum = 0.;
  for (uint32_t i = 0; i < _row; i++) {
    for (uint32_t j = 0; j < _col; j++) {
      sum += _data[i][j];
    }
  }
  return sum;
}

float Mat2D::RowSum(uint32_t i) {
  float sum = 0.;
  for (uint32_t j = 0; j < _col; j++) {
    sum += _data[i][j];
  }
  return sum;
}
float Mat2D::ColSum(uint32_t j) {
  float sum = 0.;
  for (uint32_t i = 0; i < _row; i++) {
    sum += _data[i][j];
  }
  return sum;
}

uint32_t Mat2D::Size() {
  return _row * _col;
}
#endif

#if 0
uint32_t getSize_0(Mat2D m) {
  uint32_t size = m.Row() * m.Col();
  return size;
}

float add_0(float a, float b) { return a + b; }

//=================================================

uint32_t getSize_1(Mat2D m) {
  uint32_t size = m.Row() * m.Col();
  return size;
}
uint32_t getSize_2(Mat2D m) {
  uint32_t size = m.Row() * m.Col();
  return size;
}

float add_1(float a, float b) { return a + b; }
float add_2(float a, float b) { return a + b; }
#endif

} // namespace cv