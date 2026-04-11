#include "matrix_multiplication.hpp"
#include "tensor.hpp"
#include "details.hpp"

namespace attention::matmul {
    namespace {
        void checkMatMulDimensions(const Tensor &lhs, const Tensor &rhs, const Tensor &result,
                                   const std::size_t batch_idx) {
            const auto expected_rows = lhs.rows();
            const auto expected_cols = rhs.cols();

            assert(batch_idx < lhs.batches() && details::kMsgBatchIndexOutOfRange);
            assert(result.rows() == expected_rows && details::kMsgMatMulOutputDimMismatch);
            assert(result.cols() == expected_cols && details::kMsgMatMulOutputDimMismatch);
            assert(lhs.cols() == rhs.rows() && details::kMsgMatMulInnerDimMismatch);
        }
    }

    void matMulNaive(const Tensor &lhs, const Tensor &rhs, Tensor &result, const std::size_t batch_idx) {
        const auto M = lhs.rows();
        const auto K = lhs.cols();
        const auto N = rhs.cols();

        checkMatMulDimensions(lhs, rhs, result, batch_idx);

        for (std::size_t row = 0; row < M; ++row) {
            for (std::size_t col = 0; col < N; ++col) {
                for (std::size_t k = 0; k < K; ++k) {
                    result(batch_idx, row, col) += lhs(batch_idx, row, k) * rhs(batch_idx, k, col);
                }
            }
        }
    }

    void matMulCacheOptimized(const Tensor &lhs, const Tensor &rhs, Tensor &result, const std::size_t batch_idx) {
        const auto M = lhs.rows();
        const auto K = lhs.cols();
        const auto N = rhs.cols();

        checkMatMulDimensions(lhs, rhs, result, batch_idx);

        for (std::size_t row = 0; row < M; ++row) {
            for (std::size_t k = 0; k < K; ++k) {
                for (std::size_t col = 0; col < N; ++col) {
                    result(batch_idx, row, col) += lhs(batch_idx, row, k) * rhs(batch_idx, k, col);
                }
            }
        }
    }

    void matMulTiling(const Tensor &lhs, const Tensor &rhs, Tensor &result, const std::size_t batch_idx) {
        const auto M = lhs.rows();
        const auto K = lhs.cols();
        const auto N = rhs.cols();

        auto batch_view = result.getBatchView(batch_idx);
        std::ranges::fill(batch_view, 0);

        checkMatMulDimensions(lhs, rhs, result, batch_idx);

        constexpr std::size_t BLOCK = 64;

        for (std::size_t i_block = 0; i_block < M; i_block += BLOCK) {
            const std::size_t i_end = std::min(i_block + BLOCK, M);

            for (std::size_t k_block = 0; k_block < K; k_block += BLOCK) {
                const std::size_t k_end = std::min(k_block + BLOCK, K);

                for (std::size_t j_block = 0; j_block < N; j_block += BLOCK) {
                    const std::size_t j_end = std::min(k_block + BLOCK, N);

                    for (std::size_t i = i_block; i < i_end; ++i) {
                        for (std::size_t k = k_block; k < k_end; ++k) {
                            for (std::size_t j = j_block; j < j_end; ++j) {
                                result(batch_idx, i, j) += lhs(batch_idx, i, k) * rhs(batch_idx, k, j);
                            }
                        }
                    }

                }
            }
        }
    }
}
