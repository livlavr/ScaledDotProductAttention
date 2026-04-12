#include <algorithm>

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

    void matMulTiling(const Tensor &lhs, const Tensor &rhs, Tensor &result, const std::size_t batch_idx) {
        checkMatMul(lhs, rhs, result, batch_idx);
        const auto [M, K, N] = getMatMulDimensions(lhs, rhs, result, batch_idx);
        auto [lhs_ptr, rhs_ptr, res_ptr] = getBatchPointers(lhs, rhs, result, batch_idx);

        for (std::size_t i_block = 0; i_block < M; i_block += details::kBlockSize) {
            const std::size_t i_end = std::min(i_block + details::kBlockSize, M);

            for (std::size_t k_block = 0; k_block < K; k_block += details::kBlockSize) {
                const std::size_t k_end = std::min(k_block + details::kBlockSize, K);

                for (std::size_t j_block = 0; j_block < N; j_block += details::kBlockSize) {
                    const std::size_t j_end = std::min(j_block + details::kBlockSize, N);

                    for (std::size_t row = i_block; row < i_end; ++row) {
                        for (std::size_t k = k_block; k < k_end; ++k) {
                            const float lhs_val = lhs_ptr[index(row, k, K)];

                            for (std::size_t col = j_block; col < j_end; ++col) {
                                res_ptr[index(row, col, N)] += lhs_val * rhs_ptr[index(k, col, N)];
                            }
                        }
                    }

                }
            }
        }
    }

    void matMulSIMD(const Tensor& lhs, const Tensor& rhs, Tensor& result, const std::size_t batch_idx) {
        checkMatMul(lhs, rhs, result, batch_idx);
        const auto [M, K, N] = getMatMulDimensions(lhs, rhs, result, batch_idx);

        //TODO
    }
}