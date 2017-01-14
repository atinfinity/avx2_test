#include <opencv2/core.hpp>
#include <immintrin.h>
#include <iostream>

int main(int argc, const char* argv[])
{
    bool hasAVX2 = cv::checkHardwareSupport(CV_CPU_AVX2);
    std::cout << "hasAVX2: " << (hasAVX2 ? "true" : "false") << std::endl;
    return 0;
}
