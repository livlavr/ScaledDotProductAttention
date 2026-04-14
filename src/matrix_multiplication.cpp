#include <algorithm>
#include <immintrin.h>

#include "matrix_multiplication.hpp"
#include "tensor.hpp"
#include "details.hpp"

namespace attention::matmul {

    namespace {
        struct MatMulSizes {
            const std::size_t M;
            const std::size_t K;
            const std::size_t N;
        };

        struct BatchPointers {
            const float* /*__restrict__*/ lhs_ptr; //TODO
            const float* /*__restrict__*/ rhs_ptr;
            float* /*__restrict__*/ res_ptr;
        };
        
        void checkMatMul(const Tensor &lhs, const Tensor &rhs, const Tensor &result,
                                   const std::size_t batch_idx) {
            const auto expected_rows = lhs.rows();
            const auto expected_cols = rhs.cols();
            
            assert(&lhs != &result && details::kMsgMatMulAliasingDetected);
            assert(&rhs != &result && details::kMsgMatMulAliasingDetected);

            assert(batch_idx < lhs.batches() && details::kMsgBatchIndexOutOfRange);
            assert(result.rows() == expected_rows && details::kMsgMatMulOutputDimMismatch);
            assert(result.cols() == expected_cols && details::kMsgMatMulOutputDimMismatch);
            assert(lhs.cols() == rhs.rows() && details::kMsgMatMulInnerDimMismatch);
        }

        MatMulSizes getMatMulDimensions(const Tensor &lhs, const Tensor &rhs, Tensor &result, const std::size_t batch_idx) {
            return { lhs.rows(), lhs.cols(), rhs.cols() };
        }

        BatchPointers getBatchPointers(const Tensor &lhs, const Tensor &rhs, Tensor &result, const std::size_t batch_idx) {
            auto batch_view = result.getBatchView(batch_idx);
            std::ranges::fill(batch_view, 0);

            const float* lhs_ptr = lhs.getBatchView(batch_idx).data();
            const float* rhs_ptr = rhs.getBatchView(batch_idx).data();
            float* res_ptr       = result.getBatchView(batch_idx).data();

            return { lhs_ptr, rhs_ptr, res_ptr };
        }

        std::size_t index(const std::size_t row_idx, const std::size_t col_idx, const std::size_t row_size) {
            return row_idx * row_size + col_idx;
        }
    }

    void matMulNaive(const Tensor &lhs, const Tensor &rhs, Tensor &result, const std::size_t batch_idx) {
        checkMatMul(lhs, rhs, result, batch_idx);
        auto [M, K, N] = getMatMulDimensions(lhs, rhs, result, batch_idx);

        for (std::size_t row = 0; row < M; ++row) {
            for (std::size_t col = 0; col < N; ++col) {
                for (std::size_t k = 0; k < K; ++k) {
                    result(batch_idx, row, col) += lhs(batch_idx, row, k) * rhs(batch_idx, k, col);
                }
            }
        }
    }

    void matMulDirectNaive(const Tensor &lhs, const Tensor &rhs, Tensor &result, const std::size_t batch_idx) {
        checkMatMul(lhs, rhs, result, batch_idx);
        auto [M, K, N] = getMatMulDimensions(lhs, rhs, result, batch_idx);
        auto [lhs_ptr, rhs_ptr, res_ptr] = getBatchPointers(lhs, rhs, result, batch_idx);

        for (std::size_t row = 0; row < M; ++row) {
            for (std::size_t col = 0; col < N; ++col) {
                for (std::size_t k = 0; k < K; ++k) {
                    res_ptr[index(row, col, N)] += lhs_ptr[index(row, k, K)] * rhs_ptr[index(k, col, N)];
                }
            }
        }
    }

    void matMulCacheOptimized(const Tensor &lhs, const Tensor &rhs, Tensor &result, const std::size_t batch_idx) {
        checkMatMul(lhs, rhs, result, batch_idx);
        auto [M, K, N] = getMatMulDimensions(lhs, rhs, result, batch_idx);
        auto [lhs_ptr, rhs_ptr, res_ptr] = getBatchPointers(lhs, rhs, result, batch_idx);

        for (std::size_t row = 0; row < M; ++row) {
            for (std::size_t k = 0; k < K; ++k) {
                const float lhs_val = lhs_ptr[index(row, k, K)];

                for (std::size_t col = 0; col < N; ++col) {
                    res_ptr[index(row, col, N)] += lhs_val * rhs_ptr[index(k, col, N)];
                }
            }
        }
    }

    // void matMulTiling(const Tensor &lhs, const Tensor &rhs, Tensor &result, const std::size_t batch_idx) {
    //     checkMatMul(lhs, rhs, result, batch_idx);
    //     const auto [M, K, N] = getMatMulDimensions(lhs, rhs, result, batch_idx);
    //     auto [lhs_ptr, rhs_ptr, res_ptr] = getBatchPointers(lhs, rhs, result, batch_idx);
    //
    //     for (std::size_t i_block = 0; i_block < M; i_block += details::kBlockSize) {
    //         // const std::size_t i_end = std::min(i_block + details::kBlockSize, M);
    //
    //         for (std::size_t k_block = 0; k_block < K; k_block += details::kBlockSize) {
    //             // const std::size_t k_end = std::min(k_block + details::kBlockSize, K);
    //
    //             for (std::size_t j_block = 0; j_block < N; j_block += details::kBlockSize) {
    //                 // const std::size_t j_end = std::min(j_block + details::kBlockSize, N);
    //
    //                 for (std::size_t row = i_block; row < i_block + details::kBlockSize; ++row) {
    //                     for (std::size_t k = k_block; k < k_block + details::kBlockSize; ++k) {
    //                         const float lhs_val = lhs_ptr[index(row, k, K)];
    //
    //                         for (std::size_t col = j_block; col < j_block + details::kBlockSize; ++col) {
    //                             res_ptr[index(row, col, N)] += lhs_val * rhs_ptr[index(k, col, N)];
    //                         }
    //                     }
    //                 }
    //
    //             }
    //         }
    //     }
    // }

    /*

     Изначально:
    3. Cache Optimized        | Время:   865.83 ms  | Скорость:   19.84 GFLOPS
    4. Tiling (Block=64)      | Время:   542.12 ms  | Скорость:   31.69 GFLOPS

    Performance counter stats for './cmake-build-release-remote-host/benchmark_run' (3 runs):
    12,106,216,687      cpu_core/cycles/u                                                       ( +-  0.59% )

    21,072,668,474      cpu_core/instructions/u                                                 ( +-  0.12% )

     1,980,911,777      cpu_core/branches/u                                                     ( +-  0.11% )

         4,777,390      cpu_core/branch-misses/u                                                ( +-  0.21% )

     1,116,882,636      cpu_core/cache-references/u                                             ( +-  0.24% )

       393,250,766      cpu_core/cache-misses/u                                                 ( +-  1.85% )



     Убрал min:
    3. Cache Optimized        | Время:   870.78 ms  | Скорость:   19.73 GFLOPS
    4. Tiling (Block=64)      | Время:   470.13 ms  | Скорость:   36.54 GFLOPS

     Performance counter stats for './cmake-build-release-remote-host/benchmark_run' (3 runs):
    11,545,560,503      cpu_core/cycles/u                                                       ( +-  0.61% )  (99.90%)
    17,551,842,068      cpu_core/instructions/u                                                 ( +-  0.09% )  (99.90%)
     1,171,083,205      cpu_core/branches/u                                                     ( +-  0.08% )  (99.90%)
         2,131,197      cpu_core/branch-misses/u                                                ( +-  1.95% )  (99.90%)
     1,116,379,705      cpu_core/cache-references/u                                             ( +-  0.10% )  (99.90%)
       406,667,939      cpu_core/cache-misses/u                                                 ( +-  0.51% )  (99.90%)

            2.7325 +- 0.0134 seconds time elapsed  ( +-  0.49% )


     Убрал функцию index:
    3. Cache Optimized        | Время:   879.10 ms  | Скорость:   19.54 GFLOPS
    4. Tiling (Block=64)      | Время:   472.33 ms  | Скорость:   36.37 GFLOPS


    Performance counter stats for './cmake-build-release-remote-host/benchmark_run' (5 runs):
    11,421,197,039      cpu_core/cycles/u                                                       ( +-  0.41% )  (99.58%)
    17,302,980,417      cpu_core/instructions/u                                                 ( +-  0.06% )  (99.58%)
     1,168,801,333      cpu_core/branches/u                                                     ( +-  0.16% )  (99.58%)
         1,864,464      cpu_core/branch-misses/u                                                ( +-  2.89% )  (99.58%)
     1,121,374,366      cpu_core/cache-references/u                                             ( +-  0.14% )  (99.58%)
       392,607,027      cpu_core/cache-misses/u                                                 ( +-  1.91% )  (99.58%)

            2.7184 +- 0.0129 seconds time elapsed  ( +-  0.47% )


        Pointer Bumping:
    3. Cache Optimized        | Время:   875.63 ms  | Скорость:   19.62 GFLOPS
    4. Tiling (Block=64)      | Время:   474.44 ms  | Скорость:   36.21 GFLOPS

 Performance counter stats for './cmake-build-release-remote-host/benchmark_run' (5 runs):
    11,257,154,809      cpu_core/cycles/u                                                       ( +-  0.48% )  (99.43%)
    18,590,908,419      cpu_core/instructions/u                                                 ( +-  0.08% )  (99.43%)
     1,168,713,159      cpu_core/branches/u                                                     ( +-  0.14% )  (99.43%)
           751,999      cpu_core/branch-misses/u                                                ( +-  3.78% )  (99.43%)
     1,118,091,598      cpu_core/cache-references/u                                             ( +-  0.18% )  (99.43%)
       360,796,822      cpu_core/cache-misses/u                                                 ( +-  3.07% )  (99.43%)

            2.6675 +- 0.0166 seconds time elapsed  ( +-  0.62% )


        Добавил kernel:
    3. Cache Optimized        | Время:  9251.28 ms  | Скорость:   14.86 GFLOPS
    4. Tiling (Block=64)      | Время:  3966.55 ms  | Скорость:   34.65 GFLOPS


    Performance counter stats for './cmake-build-release-remote-host/benchmark_run' (2 runs):

   112,267,390,918      cpu_core/cycles/u                                                       ( +-  1.05% )  (100.00%)
   133,454,789,225      cpu_core/instructions/u                                                 ( +-  0.03% )  (100.00%)
     6,856,583,379      cpu_core/branches/u                                                     ( +-  0.03% )  (100.00%)
        10,346,506      cpu_core/branch-misses/u                                                ( +-  0.89% )  (100.00%)
     9,152,497,926      cpu_core/cache-references/u                                             ( +-  0.00% )  (100.00%)
     7,203,317,074      cpu_core/cache-misses/u                                                 ( +-  0.22% )  (100.00%)

    */
    inline void naive_block(const float* a, const float* mb, float* c, const std::size_t N, const std::size_t K) {
        for (std::size_t row = 0; row < details::kBlockSize; ++row, c += N, a += K) {
            const float* b = mb;

            for (std::size_t k = 0; k < details::kBlockSize; ++k, b += N) {
                const float a_val = a[k];

                for (std::size_t col = 0; col < details::kBlockSize; ++col) {
                    c[col] += a_val * b[col];
                }
            }
        }
    }

    void matMulTiling(const Tensor &lhs, const Tensor &rhs, Tensor &result, const std::size_t batch_idx) {
        checkMatMul(lhs, rhs, result, batch_idx);
        const auto [M, K, N] = getMatMulDimensions(lhs, rhs, result, batch_idx);
        auto [lhs_ptr, rhs_ptr, res_ptr] = getBatchPointers(lhs, rhs, result, batch_idx);

        for (std::size_t i_block = 0; i_block < M; i_block += details::kBlockSize) {
            for (std::size_t k_block = 0; k_block < K; k_block += details::kBlockSize) {
                for (std::size_t j_block = 0; j_block < N; j_block += details::kBlockSize) {
                    const float* a = &(lhs_ptr[i_block * K + k_block]);
                    const float* b = &(rhs_ptr[k_block * N + j_block]);
                    float* c = &(res_ptr[i_block * N + j_block]);

                    naive_block(a, b, c, N, K);
                }
            }
        }
    }

    inline void simd_block(const float* a, const float* mb, float* c, const std::size_t N, const std::size_t K) {
        constexpr std::size_t kStep = 8;

        for (std::size_t row = 0; row < details::kBlockSize; ++row, c += N, a += K) {
            const float* b = mb;

            for (std::size_t k = 0; k < details::kBlockSize; ++k, b += N) {
                __m256 a_vec = _mm256_broadcast_ss(&a[k]);

                for (std::size_t col = 0; col < details::kBlockSize; col += kStep) {
                    __m256 b_vec = _mm256_load_ps(&b[col]);
                    __m256 c_vec = _mm256_load_ps(&c[col]);

                    c_vec = _mm256_fmadd_ps(a_vec, b_vec, c_vec);
                    _mm256_store_ps(&c[col], c_vec);
                }
            }
        }
    }

    void matMulSIMD(const Tensor& lhs, const Tensor& rhs, Tensor& result, const std::size_t batch_idx) {
        checkMatMul(lhs, rhs, result, batch_idx);
        const auto [M, K, N] = getMatMulDimensions(lhs, rhs, result, batch_idx);
        auto [lhs_ptr, rhs_ptr, res_ptr] = getBatchPointers(lhs, rhs, result, batch_idx);

        for (std::size_t i_block = 0; i_block < M; i_block += details::kBlockSize) {
            for (std::size_t k_block = 0; k_block < K; k_block += details::kBlockSize) {
                for (std::size_t j_block = 0; j_block < N; j_block += details::kBlockSize) {
                    const float* a = &(lhs_ptr[i_block * K + k_block]);
                    const float* b = &(rhs_ptr[k_block * N + j_block]);
                    float* c = &(res_ptr[i_block * N + j_block]);

                    simd_block(a, b, c, N, K);
                }
            }
        }
    }
}