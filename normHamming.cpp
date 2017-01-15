#include "normHamming.hpp"
#include <immintrin.h>

static const uchar popCountTable[] =
{
    0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4, 1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    1, 2, 2, 3, 2, 3, 3, 4, 2, 3, 3, 4, 3, 4, 4, 5, 2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    2, 3, 3, 4, 3, 4, 4, 5, 3, 4, 4, 5, 4, 5, 5, 6, 3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7,
    3, 4, 4, 5, 4, 5, 5, 6, 4, 5, 5, 6, 5, 6, 6, 7, 4, 5, 5, 6, 5, 6, 6, 7, 5, 6, 6, 7, 6, 7, 7, 8
};

static inline int _mm256_extract_epi32_(__m256i reg, const int i)
{
    CV_DECL_ALIGNED(32) int reg_data[8];
    CV_DbgAssert(0 <= i && i < 8);
    _mm256_store_si256((__m256i*)reg_data, reg);
    return reg_data[i];
}

int normHamming_naive(const uchar* a, int n)
{
    int i = 0;
    int result = 0;

    for (; i <= n - 4; i += 4)
        result += popCountTable[a[i]] + popCountTable[a[i + 1]] +
        popCountTable[a[i + 2]] + popCountTable[a[i + 3]];
    for (; i < n; i++)
        result += popCountTable[a[i]];
    return result;
}

int normHamming_avx2(const uchar* a, int n)
{
    int i = 0;
    int result = 0;

    bool useAVX2 = cv::checkHardwareSupport(CV_CPU_AVX2);
    if (useAVX2)
    {
        __m256i _r0 = _mm256_setzero_si256();
        __m256i _0 = _mm256_setzero_si256();
        __m256i _popcnt_table = _mm256_setr_epi8(0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4,
            0, 1, 1, 2, 1, 2, 2, 3, 1, 2, 2, 3, 2, 3, 3, 4);
        __m256i _popcnt_mask = _mm256_set1_epi8(0x0F);

        for (; i <= n - 32; i += 32)
        {
            __m256i _a0 = _mm256_loadu_si256((const __m256i*)(a + i));

            __m256i _popc0 = _mm256_shuffle_epi8(_popcnt_table, _mm256_and_si256(_a0, _popcnt_mask));
            __m256i _popc1 = _mm256_shuffle_epi8(_popcnt_table,
                _mm256_and_si256(_mm256_srli_epi16(_a0, 4), _popcnt_mask));

            _r0 = _mm256_add_epi32(_r0, _mm256_sad_epu8(_0, _mm256_add_epi8(_popc0, _popc1)));
        }
        _r0 = _mm256_add_epi32(_r0, _mm256_shuffle_epi32(_r0, 2));
        result = _mm256_extract_epi32_(_mm256_add_epi32(_r0, _mm256_permute2x128_si256(_r0, _r0, 1)), 0);
    }

    for (; i <= n - 4; i += 4)
        result += popCountTable[a[i]] + popCountTable[a[i + 1]] +
        popCountTable[a[i + 2]] + popCountTable[a[i + 3]];
    for (; i < n; i++)
        result += popCountTable[a[i]];
    return result;
}

double normHamming(cv::InputArray src_, enum NORM_HAMMING_IMPL_TYPE impl_type)
{
    cv::Mat src = src_.getMat();
    int depth = src.depth(), cn = src.channels();
    size_t len = src.total()*cn;
    uchar* data = src.ptr<uchar>();
    double result = 0.0;

    switch (impl_type)
    {
    case NORM_HAMMING_IMPL_TYPE_NAIVE:
        result = normHamming_naive(data, (int)len);
        break;
    case NORM_HAMMING_IMPL_TYPE_AVX2:
        result = normHamming_avx2(data, (int)len);
        break;
    }
    return result;
}
