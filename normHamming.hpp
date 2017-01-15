#pragma once

#include <opencv2/core.hpp>

enum NORM_HAMMING_IMPL_TYPE
{
    NORM_HAMMING_IMPL_TYPE_NAIVE = 0,
    NORM_HAMMING_IMPL_TYPE_AVX2  = 1
};

double normHamming(cv::InputArray src_, enum NORM_HAMMING_IMPL_TYPE impl_type);
