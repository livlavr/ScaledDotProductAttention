#pragma once

#include "tensor.hpp"

namespace attention::matmul {
    enum class MatMulType {
        NAIVE,
        CACHE_OPTIMIZED,
        SIMD
    };

    void matMulNaive(const Tensor &lhs, const Tensor &rhs, Tensor &result, std::size_t batch_idx);

    void matMulDirectNaive(const Tensor &lhs, const Tensor &rhs, Tensor &result, std::size_t batch_idx);

    void matMulCacheOptimized(const Tensor &lhs, const Tensor &rhs, Tensor &result, std::size_t batch_idx);

    void matMulTiling(const Tensor &lhs, const Tensor &rhs, Tensor &result, std::size_t batch_idx);

    void matMulSIMD(const Tensor &lhs, const Tensor &rhs, Tensor &result, std::size_t batch_idx);

    void matMulSafe(const Tensor &lhs, const Tensor &rhs, Tensor &result, std::size_t batch_idx);

    void matMul(MatMulType type, const Tensor &lhs, const Tensor &rhs, Tensor &result, std::size_t batch_idx);
}
