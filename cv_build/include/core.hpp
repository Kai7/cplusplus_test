#ifndef _AI_CORE_HPP_
#define _AI_CORE_HPP_

// #include "cv/core.hpp"
#include "opencv2/core.hpp"

class AIMat {
public:
  AIMat();
  AIMat(cv::Mat &src);
  ~AIMat();

  float Trace();

private:
  cv::Mat _m;
};

#endif