#include "gelu_omp.h"

#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

template <typename Func>
double Benchmark(Func&& f, int iterations = 5) {
    double best = 1e100;

    for (int i = 0; i < iterations; ++i) {
        auto start = std::chrono::high_resolution_clock::now();

        auto result = f();

        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> elapsed = end - start;
        best = std::min(best, elapsed.count());

        volatile float sink = result[result.size() / 2];
        (void)sink;
    }

    return best;
}

int main() {
    constexpr size_t N = 100000000;

    std::vector<float> input(N);

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(-5.f, 5.f);

    for (auto& x : input) {
        x = dist(rng);
    }

    std::cout << "Input size: " << N << "\n\n";

    auto ref_time = Benchmark([&]() {
        return GeluRef(input);
    });

    auto omp_time = Benchmark([&]() {
        return GeluOMP(input);
    });

    auto ref = GeluRef(input);
    auto omp = GeluOMP(input);

    auto max_error = 0.0f;

    for (size_t i = 0; i < input.size(); ++i) {
        max_error = std::max(
            max_error,
            std::abs(ref[i] - omp[i]));
    }

    std::cout << "Reference : " << ref_time << " s\n";
    std::cout << "OpenMP    : " << omp_time << " s\n";
    std::cout << "Speedup   : " << ref_time / omp_time << "x\n";
    std::cout << "Max error : " << max_error << '\n';

    return 0;
}