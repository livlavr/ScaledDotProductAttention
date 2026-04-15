#include <cassert>
#include <span>
#include <stdexcept>
#include <algorithm>
#include <string>
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
            const float *lhs_ptr;
            const float *rhs_ptr;
            float *res_ptr;
        };

        void validateMatMul(const Tensor &lhs, const Tensor &rhs, const Tensor &result,
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

        bool isTilingCompatible(const std::size_t M, const std::size_t K, const std::size_t N) {
            constexpr std::size_t B = details::kBlockSize;
            return (M % B == 0 && K % B == 0 && N % B == 0);
        }

        void validateTilingDimensions(const std::size_t M, const std::size_t K, const std::size_t N) {
            if (not isTilingCompatible(M, K, N)) {
                throw std::invalid_argument("Matrix dimensions are not compatible with Tiling (must be multiples of "
                                            + std::to_string(details::kBlockSize) + ")");
            }
        }

        MatMulSizes getMatMulDimensions(const Tensor &lhs, const Tensor &rhs) {
            return {lhs.rows(), lhs.cols(), rhs.cols()};
        }

        BatchPointers getBatchPointers(const Tensor &lhs, const Tensor &rhs, Tensor &result,
                                       const std::size_t batch_idx) {
            auto batch_view = result.getBatchView(batch_idx);
            std::ranges::fill(batch_view, 0);

            const float *lhs_ptr = lhs.getBatchView(batch_idx).data();
            const float *rhs_ptr = rhs.getBatchView(batch_idx).data();
            float *res_ptr = result.getBatchView(batch_idx).data();

            return {lhs_ptr, rhs_ptr, res_ptr};
        }

        std::size_t index(const std::size_t row_idx, const std::size_t col_idx, const std::size_t row_size) {
            return row_idx * row_size + col_idx;
        }
    }

    void matMulNaive(const Tensor &lhs, const Tensor &rhs, Tensor &result, const std::size_t batch_idx) {
        validateMatMul(lhs, rhs, result, batch_idx);
        auto [M, K, N] = getMatMulDimensions(lhs, rhs);

        for (std::size_t row = 0; row < M; ++row) {
            for (std::size_t col = 0; col < N; ++col) {
                for (std::size_t k = 0; k < K; ++k) {
                    result(batch_idx, row, col) += lhs(batch_idx, row, k) * rhs(batch_idx, k, col);
                }
            }
        }
    }

    void matMulDirectNaive(const Tensor &lhs, const Tensor &rhs, Tensor &result, const std::size_t batch_idx) {
        validateMatMul(lhs, rhs, result, batch_idx);
        auto [M, K, N] = getMatMulDimensions(lhs, rhs);
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
        validateMatMul(lhs, rhs, result, batch_idx);
        auto [M, K, N] = getMatMulDimensions(lhs, rhs);
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

    inline void naive_block(const float *a, const float *mb, float *c, const std::size_t N, const std::size_t K) {
        for (std::size_t row = 0; row < details::kBlockSize; ++row, c += N, a += K) {
            const float *b = mb;

            for (std::size_t k = 0; k < details::kBlockSize; ++k, b += N) {
                const float a_val = a[k];

                for (std::size_t col = 0; col < details::kBlockSize; ++col) {
                    c[col] += a_val * b[col];
                }
            }
        }
    }

    void matMulTiling(const Tensor &lhs, const Tensor &rhs, Tensor &result, const std::size_t batch_idx) {
        validateMatMul(lhs, rhs, result, batch_idx);
        const auto [M, K, N] = getMatMulDimensions(lhs, rhs);
        auto [lhs_ptr, rhs_ptr, res_ptr] = getBatchPointers(lhs, rhs, result, batch_idx);
        validateTilingDimensions(M, K, N);

        for (std::size_t i_block = 0; i_block < M; i_block += details::kBlockSize) {
            for (std::size_t k_block = 0; k_block < K; k_block += details::kBlockSize) {
                for (std::size_t j_block = 0; j_block < N; j_block += details::kBlockSize) {
                    const float *a = &(lhs_ptr[i_block * K + k_block]);
                    const float *b = &(rhs_ptr[k_block * N + j_block]);
                    float *c = &(res_ptr[i_block * N + j_block]);

                    naive_block(a, b, c, N, K);
                }
            }
        }
    }

    inline void simd_block(const float *a, const float *mb, float *c, const std::size_t N, const std::size_t K) {
        for (std::size_t row = 0; row < details::kBlockSize; ++row, c += N, a += K) {
            const float *b = mb;

            for (std::size_t k = 0; k < details::kBlockSize; ++k, b += N) {
                constexpr std::size_t kStep = 8;
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

    void matMulSIMD(const Tensor &lhs, const Tensor &rhs, Tensor &result, const std::size_t batch_idx) {
        validateMatMul(lhs, rhs, result, batch_idx);
        const auto [M, K, N] = getMatMulDimensions(lhs, rhs);
        auto [lhs_ptr, rhs_ptr, res_ptr] = getBatchPointers(lhs, rhs, result, batch_idx);
        validateTilingDimensions(M, K, N);

        for (std::size_t i_block = 0; i_block < M; i_block += details::kBlockSize) {
            for (std::size_t k_block = 0; k_block < K; k_block += details::kBlockSize) {
                for (std::size_t j_block = 0; j_block < N; j_block += details::kBlockSize) {
                    const float *a = &(lhs_ptr[i_block * K + k_block]);
                    const float *b = &(rhs_ptr[k_block * N + j_block]);
                    float *c = &(res_ptr[i_block * N + j_block]);

                    simd_block(a, b, c, N, K);
                }
            }
        }
    }

    void matMulSafe(const Tensor &lhs, const Tensor &rhs, Tensor &result, const std::size_t batch_idx) {
        if (isTilingCompatible(lhs.cols(), lhs.rows(), rhs.cols())) {
            matMulTiling(lhs, rhs, result, batch_idx);
        } else {
            matMulCacheOptimized(lhs, rhs, result, batch_idx);
        }
    }

    void matMul(const MatMulType type, const Tensor &lhs, const Tensor &rhs, Tensor &result, std::size_t batch_idx) {
        switch (type) {
            case MatMulType::NAIVE:
                matMulNaive(lhs, rhs, result, batch_idx);
                break;
            case MatMulType::CACHE_OPTIMIZED:
                matMulSafe(lhs, rhs, result, batch_idx);
                break;
            case MatMulType::SIMD:
                matMulSIMD(lhs, rhs, result, batch_idx);
                break;
            default:
                matMulNaive(lhs, rhs, result, batch_idx);
        }
    }
}
