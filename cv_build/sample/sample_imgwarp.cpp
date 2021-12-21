#include "stdio.h"
#include "string.h"

#include "core.hpp"
#include "imgwarp.hpp"

#include "opencv2/core.hpp"
#include "opencv2/imgcodecs.hpp"

int main(){
    printf("main()\n");

    std::string img_path = "images/sample_01.jpg";

    cv::Mat img = cv::imread(img_path.c_str());

    printf("Image Size: (%d, %d)\n", img.rows, img.cols);

    ai::getPerspectiveTransform();

    return 0;
}