#include "core.hpp"

AIMat::AIMat() {}

AIMat::~AIMat() {}

AIMat::AIMat(cv::Mat &src) { _m = src; }

float AIMat::Trace() { return 0.; }