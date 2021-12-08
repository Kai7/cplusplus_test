#include "core.hpp"

AIMat2D::AIMat2D() : _m() {}

AIMat2D::~AIMat2D() {}

AIMat2D::AIMat2D(uint32_t row, uint32_t col) : _m(row, col) {}

AIMat2D::AIMat2D(uint32_t row, uint32_t col, float **data)
    : _m(row, col, data) {}

float AIMat2D::Trace() {
  if (_m.Row() != _m.Col())
    return 0.;
  float trace = 0.;
  for (uint32_t i = 0; i < _m.Row(); i++) {
    trace += _m.at(i, i);
  }
  return trace;
}