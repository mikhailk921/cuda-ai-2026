#include "gelu_omp.h"
#include <vector>
#include <cmath>
#include <omp.h>
#include <immintrin.h>


#pragma GCC optimize("O3")
#pragma omp
#pragma simd


#pragma GCC target("avx2,fma")

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

inline __m256 exp256_ps(__m256 x) {
    __m256   exp_hi        = _mm256_set1_ps(88.3762626647949f);
    __m256   exp_lo        = _mm256_set1_ps(-88.3762626647949f);

    __m256   cephes_LOG2EF = _mm256_set1_ps(1.44269504088896341f);
    __m256   cephes_exp_C1 = _mm256_set1_ps(0.693359375f);
    __m256   cephes_exp_C2 = _mm256_set1_ps(-2.12194440e-4f);

    __m256   cephes_exp_p0 = _mm256_set1_ps(1.9875691500E-4f);
    __m256   cephes_exp_p1 = _mm256_set1_ps(1.3981999507E-3f);
    __m256   cephes_exp_p2 = _mm256_set1_ps(8.3334519073E-3f);
    __m256   cephes_exp_p3 = _mm256_set1_ps(4.1665795894E-2f);
    __m256   cephes_exp_p4 = _mm256_set1_ps(1.6666665459E-1f);
    __m256   cephes_exp_p5 = _mm256_set1_ps(5.0000001201E-1f);
    __m256   one           = _mm256_set1_ps(1.0f);

    x = _mm256_min_ps(x, exp_hi);
    x = _mm256_max_ps(x, exp_lo);

    __m256 fx = _mm256_mul_ps(x, cephes_LOG2EF);
    fx = _mm256_add_ps(fx, _mm256_set1_ps(0.5f));
    __m256 tmp = _mm256_floor_ps(fx);
    __m256 mask = _mm256_cmp_ps(tmp, fx, _CMP_GT_OS);    
    mask = _mm256_and_ps(mask, one);
    fx = _mm256_sub_ps(tmp, mask);
    tmp = _mm256_mul_ps(fx, cephes_exp_C1);
    __m256 z = _mm256_mul_ps(fx, cephes_exp_C2);
    x = _mm256_sub_ps(x, tmp);
    x = _mm256_sub_ps(x, z);
    z = _mm256_mul_ps(x, x);

    __m256 y = cephes_exp_p0;
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, cephes_exp_p1);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, cephes_exp_p2);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, cephes_exp_p3);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, cephes_exp_p4);
    y = _mm256_mul_ps(y, x);
    y = _mm256_add_ps(y, cephes_exp_p5);
    y = _mm256_mul_ps(y, z);
    y = _mm256_add_ps(y, x);
    y = _mm256_add_ps(y, one);

    __m256i imm0 = _mm256_cvttps_epi32(fx);
    imm0 = _mm256_add_epi32(imm0, _mm256_set1_epi32(0x7f));
    imm0 = _mm256_slli_epi32(imm0, 23);
    __m256 pow2n = _mm256_castsi256_ps(imm0);
    y = _mm256_mul_ps(y, pow2n);
    return y;
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t size = input.size();
    std::vector<float> output(size);

    const float* xp = input.data();
    float* yp = output.data();

    constexpr float c1 = 0.044715f;

    const float c2 = 2.0f * std::sqrt(2.0f / static_cast<float>(M_PI));

    const __m256 coeff1 = _mm256_set1_ps(c1);
    const __m256 coeff2 = _mm256_set1_ps(c2);
    const __m256 one = _mm256_set1_ps(1.0f);

    constexpr size_t VecSize = 8;

    size_t vec_size_end = size - (size % VecSize);

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < vec_size_end; i += VecSize) {
        __m256 x = _mm256_loadu_ps(xp + i);
        
        // k = 1.0 + c1 * x * x
        __m256 x2 = _mm256_mul_ps(x, x);
        __m256 k = _mm256_fmadd_ps(coeff1, x2, one);
        
        __m256 arg = _mm256_mul_ps(_mm256_mul_ps(coeff2, k), x);
        
        __m256 expRes = exp256_ps(arg);
        __m256 denom = _mm256_add_ps(expRes, one);
        __m256 inv_denom = _mm256_div_ps(one, denom);
        __m256 bracket = _mm256_sub_ps(one, inv_denom);
        __m256 res = _mm256_mul_ps(x, bracket);

        _mm256_storeu_ps(yp + i, res);
    }


    #pragma omp parallel for schedule(static)
    for (size_t j = vec_size_end; j < size; ++j) {
        float x = input[j];
        float inner = c2 * x * (1.0f + c1 * x * x);
        output[j] = x * (1.0f - 1.0f / (std::exp(inner) + 1.0f));
    }

    return output;
}