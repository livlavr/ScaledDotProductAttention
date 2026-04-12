#include <iostream>
#include <chrono>
#include <vector>
#include <random>
#include <iomanip>
#include <string>
#include <functional>

#include "tensor.hpp"
#include "matrix_multiplication.hpp"

using namespace attention;
using namespace attention::matmul;

namespace {
    void fill_random(Tensor& t) {
        std::mt19937 gen(42);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (std::size_t i = 0; i < t.size(); ++i) {
            t.data()[i] = dist(gen);
        }
    }

    void run_benchmark(const std::string& name, std::size_t N, int iterations,
                       const std::function<void()>& func) {
        for (int i = 0; i < std::min(3, iterations); ++i) {
            func();
        }

        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < iterations; ++i) {
            func();
        }
        auto end = std::chrono::steady_clock::now();

        std::chrono::duration<double> diff = end - start;
        double avg_time_sec = diff.count() / iterations;
        double avg_time_ms = avg_time_sec * 1000.0;

        double gflops = (2.0 * N * N * N) / (avg_time_sec * 1e9);

        std::cout << std::left << std::setw(25) << name
                  << " | Время: " << std::right << std::setw(8) << std::fixed << std::setprecision(2) << avg_time_ms << " ms "
                  << " | Скорость: " << std::setw(7) << std::fixed << std::setprecision(2) << gflops << " GFLOPS\n";
    }
}

int main() {
    std::cout << "==============================================================\n";
    std::cout << "        BENCHMARK: ATTENTION MATRIX MULTIPLICATION            \n";
    std::cout << "==============================================================\n\n";

    // std::vector<std::size_t> sizes = {128, 256, 512, 1024};
    std::vector<std::size_t> sizes = {4096};

    for (std::size_t N : sizes) {
        std::cout << "--------------------------------------------------------------\n";
        std::cout << " Размер матрицы: " << N << " x " << N << " (Batch = 1)\n";
        std::cout << "--------------------------------------------------------------\n";

        Tensor A(1, N, N);
        Tensor B(1, N, N);
        Tensor C(1, N, N);

        fill_random(A);
        fill_random(B);

        // int iters_fast = (N >= 1024) ? 5 : 20;
        // int iters_slow = (N >= 512) ? 1 : 10;

        int iters_fast = 1;

        // run_benchmark("1. Naive (operator())", N, iters_slow, [&]() { matMulNaive(A, B, C, 0); });
        // run_benchmark("2. Direct Naive (Ptr)", N, iters_slow, [&]() { matMulDirectNaive(A, B, C, 0); });
        run_benchmark("3. Cache Optimized",    N, iters_fast, [&]() { matMulCacheOptimized(A, B, C, 0); });
        run_benchmark("4. Tiling (Block=64)",  N, iters_fast, [&]() { matMulTiling(A, B, C, 0); });

        std::cout << "\n";
    }

    return 0;
}