#include "convertScale.hpp"
#include <immintrin.h>

void
cvtScale_naive(const short* src, size_t sstep,
    int* dst, size_t dstep, cv::Size size,
    float scale, float shift)
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    for (; size.height--; src += sstep, dst += dstep)
    {
        int x = 0;
        for (; x < size.width; x++)
            dst[x] = cv::saturate_cast<int>(src[x] * scale + shift);
    }
}

void
cvtScale_avx2( const short* src, size_t sstep,
           int* dst, size_t dstep, cv::Size size,
           float scale, float shift )
{
    sstep /= sizeof(src[0]);
    dstep /= sizeof(dst[0]);

    bool useAVX2 = cv::checkHardwareSupport(CV_CPU_AVX2);
    for( ; size.height--; src += sstep, dst += dstep )
    {
        int x = 0;

        if (useAVX2)
        {
            __m256 scale256 = _mm256_set1_ps(scale);
            __m256 shift256 = _mm256_set1_ps(shift);
            const int shuffle = 0xD8;

            for ( ; x <= size.width - 16; x += 16)
            {
                __m256i v_src = _mm256_loadu_si256((const __m256i *)(src + x));
                v_src = _mm256_permute4x64_epi64(v_src, shuffle);
                __m256i v_src_lo = _mm256_srai_epi32(_mm256_unpacklo_epi16(v_src, v_src), 16);
                __m256i v_src_hi = _mm256_srai_epi32(_mm256_unpackhi_epi16(v_src, v_src), 16);
                __m256 v_dst0 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(v_src_lo), scale256), shift256);
                __m256 v_dst1 = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(v_src_hi), scale256), shift256);
                _mm256_storeu_si256((__m256i *)(dst + x), _mm256_cvtps_epi32(v_dst0));
                _mm256_storeu_si256((__m256i *)(dst + x + 8), _mm256_cvtps_epi32(v_dst1));
            }
        }

        for(; x < size.width; x++ )
            dst[x] = cv::saturate_cast<int>(src[x]*scale + shift);
    }
}

void convertTo(cv::InputArray src_, cv::OutputArray dst_, int type_, double alpha, double beta, enum CONVERT_SCALE_IMPL_TYPE impl_type)
{
    bool noScale = fabs(alpha - 1) < DBL_EPSILON && fabs(beta) < DBL_EPSILON;
    int sdepth = src_.depth(), ddepth = CV_MAT_DEPTH(type_);

    cv::Mat src = src_.getMat();
    int cn = src_.channels();

    dst_.create(src.size(), type_);
    cv::Mat dst = dst_.getMat();
    cv::Size sz = getContinuousSize(src, dst, cn);

    switch (impl_type)
    {
    case CONVERT_SCALE_IMPL_TYPE_NAIVE:
        cvtScale_naive(src.ptr<short>(), src.step, dst.ptr<int>(), dst.step, sz, (float)alpha, (float)beta);
        break;
    case CONVERT_SCALE_IMPL_TYPE_AVX2:
        cvtScale_avx2(src.ptr<short>(), src.step, dst.ptr<int>(), dst.step, sz, (float)alpha, (float)beta);
        break;
    }
}
