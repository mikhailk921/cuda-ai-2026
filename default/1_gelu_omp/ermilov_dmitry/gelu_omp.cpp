
#include "gelu_omp.h"

#include <cmath>
#include <omp.h>


constexpr float SQRT_2_OVER_PI =
    std::sqrt(2.0f / static_cast<float>(M_PI));

std::vector<float> GeluRef(const std::vector<float>& input) {
    std::vector<float> output(input.size());

    for (size_t i = 0; i < input.size(); ++i) {
        const float & x = input[i];
        output[i] = 0.5f * x *  (1.f + std::tanh(SQRT_2_OVER_PI * (x + 0.044715f * x * x * x)));
    }

    return output;
}


inline float fastTanh(float x) {
    return 1.f - (2.f / (1.f + std::exp(2.f * x)));
}

std::vector<float> GeluOMP(const std::vector<float>& input) {
    std::vector<float> result(input.size());

    #pragma omp parallel for simd
    for (size_t i = 0; i < input.size(); ++i) {
        const float x = input[i];
        const float x3 =  x * x * x;
        const float tanh_val = fastTanh(SQRT_2_OVER_PI * (x + 0.044715f * x3));
        result[i] = 0.5f * x *  (1.f + tanh_val);
    }
    return result;
}