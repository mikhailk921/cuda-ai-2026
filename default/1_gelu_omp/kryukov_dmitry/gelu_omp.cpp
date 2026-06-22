#include "gelu_omp.h"
#include <vector>
#include <cmath>
#include <stdint.h>

#define NO_INIT_VECTOR

#ifdef NO_INIT_VECTOR

//
// std::vector<float> v(n) zero-fills n elements, then the OMP loop overwrites the zeros.
// That single memset accounts for ~75% of time in the CPU pipeline (on my hardware).
// The proper fix would be:
//   1. Do not use std::vector at all
//   2. Override the allocator
// But I had to use this to keep given signature.
// I am fully aware this should never appear in production code.
//

template<typename T>
struct vec_ptrs { T* start; T* finish; T* end; };

template<typename T>
static void set_vec_size(std::vector<T>& v, size_t n) {
    reinterpret_cast<vec_ptrs<T>&>(v).finish = v.data() + n;
}
#endif

constexpr float K = 0.044715f;
constexpr float COEFF = 0.7978845608028654f;

#define FASTAPPROX

#ifdef FASTAPPROX
    /// from https://github.com/romeric/fastapprox/blob/master/fastapprox/src/fastexp.h
    // Note: This approximation degrades the MAE from 1e-9 to 1e-6.
    static inline float
    fastpow2 (float p) {
    float offset = (p < 0) ? 1.0f : 0.0f;
    float clipp = (p < -126) ? -126.0f : p;
    int w = clipp;
    float z = clipp - w + offset;
    union { uint32_t i; float f; } v = { static_cast<uint32_t> ( (1 << 23) * (clipp + 121.2740575f + 27.7280233f / (4.84252568f - z) - 1.49012907f * z) ) };

    return v.f;
    }

    static inline float
    fastexp (float p) {
    return fastpow2 (1.442695040f * p);
    }

    inline float exp_tanh(float x) {
        float exp_x = fastexp(-2.0f * std::fabs(x));
        return std::copysign((1.0f - exp_x) / (1.0f + exp_x), x);
    }

#else
    inline float exp_tanh(float x) {
    float exp_x = std::exp(2.0f * std::copysign(std::fmin(std::fabs(x), 9.0f), x));
    return (exp_x - 1.0f) / (exp_x + 1.0f);
    }
#endif

std::vector<float> GeluSEQ(const std::vector<float>& input) {
    std::vector<float> result(input.size());
    for (size_t i = 0; i < input.size(); ++i) {
        result[i] = 0.5f * input[i] * (1.0f + std::tanh(COEFF * (input[i] + K * input[i] * input[i] * input[i])));
    }
    return result;
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    const size_t n = input.size();
#ifdef NO_INIT_VECTOR
    std::vector<float> result;
    result.reserve(n);
    float *__restrict p_out = result.data();
    const float *__restrict p_in = input.data();
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; ++i) {
        p_out[i] = 0.5f * p_in[i] * (1.0f + exp_tanh(COEFF * (p_in[i]  + K * p_in[i] * p_in[i] * p_in[i])));
    }
    set_vec_size(result, n);
#else
    std::vector<float> result(n);
    float *__restrict p_out = result.data();
    const float *__restrict p_in = input.data();
    #pragma omp parallel for simd schedule(static)
    for (size_t i = 0; i < n; ++i) {
        p_out[i] = 0.5f * p_in[i] * (1.0f + exp_tanh(COEFF * (p_in[i]  + K * p_in[i] * p_in[i] * p_in[i])));
    }
#endif
    return result;
}
