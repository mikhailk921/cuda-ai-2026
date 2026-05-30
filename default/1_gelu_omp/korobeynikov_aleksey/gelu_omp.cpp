#include "gelu_omp.h"

#include <omp.h>
#include <cmath>


std::vector<float> GeluOMP(const std::vector<float>& input) {
    int input_size = input.size();
    std::vector<float> result(input_size);

    // compute constant
    float sqrt_constant = std::sqrt(2.0 / M_PI);

    #pragma omp parallel for
    for (int i = 0; i < input_size; ++i) {
        float x = input[i];
        result[i] = 0.5f * x * (1.0f + std::tanh(sqrt_constant * (x + 0.044715f * x*x*x)));
    }

    return result;
}
