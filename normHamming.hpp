#pragma once

#include <opencv2/core.hpp>

enum IMPL_TYPE
{
    IMPL_TYPE_NAIVE = 0,
    IMPL_TYPE_AVX2  = 1
};

double normHamming(cv::InputArray src_, enum IMPL_TYPE impl_type);
