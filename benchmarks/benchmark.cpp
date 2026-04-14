#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <string>
#include <functional>
#include <algorithm>

#include "tensor.hpp"
#include "matrix_multiplication.hpp"
#include "attention.hpp"

using namespace attention;
using namespace attention::matmul;

namespace {
    void fill_random(Tensor &tensor) {
        std::mt19937 gen(71);
        std::uniform_real_distribution<float> dist(-100.0f, 100.0f);

        std::ranges::generate(tensor, [&]() { return dist(gen); });
    }

    void run_benchmark(const std::string &name, const std::size_t N, const int iterations,
                       const std::function<void()> &func) {
        for (int i = 0; i < std::min(5, iterations); ++i) {
            func();
        }

        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) {
            func();
        }
        const auto end = std::chrono::steady_clock::now();

        const std::chrono::duration<double> diff = end - start;
        const double avg_time_sec = diff.count() / iterations;
        const double avg_time_ms = avg_time_sec * 1000.0;

        const double gflops = (2.0 * N * N * N) / (avg_time_sec * 1e9);

        std::cout << std::left << std::setw(25) << name
                << " | Время: " << std::right << std::setw(10) << std::fixed << std::setprecision(2) << avg_time_ms <<
                " ms "
                << " | Скорость: " << std::setw(6) << std::fixed << std::setprecision(2) << gflops << " GFLOPS\n";
    }
}

int main() {
    std::cout << "==============================================================\n";
    std::cout << "               BENCHMARK: MATRIX MULTIPLICATION               \n";
    std::cout << "==============================================================\n";

    std::vector<std::size_t> sizes = {4000};

    for (const std::size_t N: sizes) {
        std::cout << "--------------------------------------------------------------\n";
        std::cout << " Размер матрицы: " << N << " x " << N << " (Batch = 1)\n";
        std::cout << "--------------------------------------------------------------\n";

        Tensor A(1, N, N);
        Tensor B(1, N, N);
        Tensor C(1, N, N);

        fill_random(A);
        fill_random(B);

        const int iters_fast = (N >= 1024) ? 2 : 10;
        const int iters_slow = (N >= 512) ? 2 : 10;

        run_benchmark("1. Naive (operator())", N, iters_slow, [&]() { matMulNaive(A, B, C, 0); });
        run_benchmark("2. Direct Naive (Ptr)", N, iters_slow, [&]() { matMulDirectNaive(A, B, C, 0); });
        run_benchmark("3. Cache Optimized", N, iters_fast, [&]() { matMulCacheOptimized(A, B, C, 0); });
        run_benchmark("4. Tiling (Block=160)", N, iters_fast, [&]() { matMulTiling(A, B, C, 0); });
        run_benchmark("5. SIMD", N, iters_fast, [&]() { matMulSIMD(A, B, C, 0); });

        std::cout << "\n";

        std::cout << "==============================================================\n";
        std::cout << "                     BENCHMARK: ATTENTION                     \n";
        std::cout << "==============================================================\n";

        fill_random(C);

        run_benchmark("1. Attention(): Naive", N, iters_fast, [&]() {
            volatile auto res = attentionWithMatmul(A, B, C, MatMulType::NAIVE);
        });

        run_benchmark("2. Attention(): Cache optimized", N, iters_fast, [&]() {
            volatile auto res = attentionWithMatmul(A, B, C, MatMulType::CACHE_OPTIMIZED);
        });

        run_benchmark("3. Attention(): SIMD", N, iters_fast, [&]() {
            volatile auto res = attentionWithMatmul(A, B, C, MatMulType::SIMD);
        });
    }

    return 0;
}
