#pragma once

#include <opencv2/core.hpp>

enum CONVERT_SCALE_IMPL_TYPE
{
    CONVERT_SCALE_IMPL_TYPE_NAIVE = 0,
    CONVERT_SCALE_IMPL_TYPE_AVX2 = 1
};

#define CV_MAT_DEPTH(flags)     ((flags) & CV_MAT_DEPTH_MASK)

inline cv::Size getContinuousSize_(int flags, int cols, int rows, int widthScale)
{
    int64 sz = (int64)cols * rows * widthScale;
    return (flags & cv::Mat::CONTINUOUS_FLAG) != 0 &&
        (int)sz == sz ? cv::Size((int)sz, 1) : cv::Size(cols * widthScale, rows);
}

inline cv::Size getContinuousSize(const cv::Mat& m1, const cv::Mat& m2, int widthScale = 1)
{
    return getContinuousSize_(m1.flags & m2.flags,
        m1.cols, m1.rows, widthScale);
}

void convertTo(cv::InputArray src_, cv::OutputArray dst_, int type_, double alpha, double beta, enum CONVERT_SCALE_IMPL_TYPE impl_type);
