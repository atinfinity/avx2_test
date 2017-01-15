#include "normHamming.hpp"
#include "convertScale.hpp"
#include <opencv2/core.hpp>
#include <iostream>
#include <cmath>

const cv::Size sz1080p = cv::Size(1920, 1080); // Full-HD
const cv::Size sz2160p = cv::Size(3840, 2160); // 4K
const cv::Size sz4320p = cv::Size(7680, 4320); // 8K

double launch_normHamming
(
    const cv::Mat& src,
    enum NORM_HAMMING_IMPL_TYPE impl_type,
    const int loop_num
)
{
    cv::TickMeter tm;
    double time = 0.0;
    for (int i = 0; i <= loop_num; i++) {
        tm.reset();
        tm.start();
        normHamming(src, impl_type);
        tm.stop();
        time += (i > 0) ? (tm.getTimeMilli()) : 0;
    }
    time /= loop_num;

    return time;
}

double launch_convertTo
(
    const cv::Mat& src,
    cv::Mat& dst,
    int type,
    double alpha,
    double beta, 
    enum CONVERT_SCALE_IMPL_TYPE impl_type,
    const int loop_num
)
{
    cv::TickMeter tm;
    double time = 0.0;
    for (int i = 0; i <= loop_num; i++) {
        tm.reset();
        tm.start();
        convertTo(src, dst, type, alpha, beta, impl_type);
        tm.stop();
        time += (i > 0) ? (tm.getTimeMilli()) : 0;
    }
    time /= loop_num;

    return time;
}

int main(int argc, const char* argv[])
{
    bool hasAVX2 = cv::checkHardwareSupport(CV_CPU_AVX2);
    std::cout << "hasAVX2: " << (hasAVX2 ? "true" : "false") << std::endl;

    cv::Size sz = sz4320p;
    std::cout << "size: " << sz << std::endl << std::endl;

    cv::Mat src(sz, CV_8UC1, cv::Scalar(0));
    cv::randu(src, cv::Scalar(0), cv::Scalar(255));

    // verification
    std::cout << "[normHamming]" << std::endl;
    double result_naive = normHamming(src, NORM_HAMMING_IMPL_TYPE_NAIVE);
    double result_avx2  = normHamming(src, NORM_HAMMING_IMPL_TYPE_AVX2);
    if (fabs(result_avx2 - result_naive) > 0)
    {
        std::cout << "verify: failed." << std::endl;
        return -1;
    }
    else
    {
        std::cout << "verify: passed." << std::endl << std::endl;
    }

    const int loop_num = 10;
    double time_naive = launch_normHamming(src, NORM_HAMMING_IMPL_TYPE_NAIVE, loop_num);
    double time_avx2  = launch_normHamming(src, NORM_HAMMING_IMPL_TYPE_AVX2,  loop_num);
    std::cout << "Naive: " << time_naive << " ms." << std::endl;
    std::cout << "AVX2: " << time_avx2 << " ms." << std::endl << std::endl;

    cv::Mat src2(sz, CV_16SC1, cv::Scalar(0));
    cv::randu(src2, cv::Scalar(0), cv::Scalar(255));

    // verification
    std::cout << "[cvtScale]" << std::endl;
    cv::Mat dst_naive(sz, CV_32SC1, cv::Scalar(0));
    cv::Mat dst_avx2(sz, CV_32SC1, cv::Scalar(0));
    cv::Mat diff(sz, CV_32SC1, cv::Scalar(0));
    convertTo(src2, dst_naive, dst_naive.depth(), 2.0, 1.0, CONVERT_SCALE_IMPL_TYPE_NAIVE);
    convertTo(src2, dst_avx2, dst_avx2.depth(), 2.0, 1.0, CONVERT_SCALE_IMPL_TYPE_AVX2);
    cv::absdiff(dst_naive, dst_avx2, diff);
    if (cv::countNonZero(diff) > 0)
    {
        std::cout << "verify: failed." << std::endl;
        return -1;
    }
    else
    {
        std::cout << "verify: passed." << std::endl << std::endl;
    }

    time_naive = launch_convertTo(src2, dst_naive, dst_naive.depth(), 2.0, 1.0, CONVERT_SCALE_IMPL_TYPE_NAIVE, loop_num);
    time_avx2 = launch_convertTo(src2, dst_avx2, dst_avx2.depth(), 2.0, 1.0, CONVERT_SCALE_IMPL_TYPE_AVX2, loop_num);
    std::cout << "Naive: " << time_naive << " ms." << std::endl;
    std::cout << "AVX2: " << time_avx2 << " ms." << std::endl;

    return 0;
}
