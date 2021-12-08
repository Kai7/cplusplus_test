#include "core.hpp"
#include <cstring>
#include "assert.h" 

Mat2D::Mat2D(): _row(0), _col(0), _data(nullptr) {}

Mat2D::Mat2D(uint32_t row, uint32_t col): _row(row), _col(col) {
  _data = new float*[row];
  for (uint32_t i = 0; i < row; i++){
    _data[i] = new float[col];
  }
}

Mat2D::Mat2D(uint32_t row, uint32_t col, float **data): _row(row), _col(col) {
  _data = new float*[row];
  for (uint32_t i = 0; i < row; i++){
    _data[i] = new float[col];
    memcpy(_data[i], data[i], sizeof(float) * col);
  }
}

Mat2D::~Mat2D(){
  for (uint32_t i = 0; i < _row; i++){
    delete [] _data[i];
  }
  delete [] _data;
  _data = nullptr;
  _row = 0;
  _col = 0;
}

uint32_t Mat2D::Col(){
  return _col;
}

uint32_t Mat2D::Row(){
  return _row;
}

float Mat2D::at(uint32_t i, uint32_t j){
  assert(i < _row && j < _col);
  return _data[i][j];
}

float Mat2D::Sum(){
  float sum = 0.;
  for (uint32_t i = 0; i < _row; i++){
    for (uint32_t j = 0; j < _col; j++){
      sum += _data[i][j];
    }
  }
  return sum;
}