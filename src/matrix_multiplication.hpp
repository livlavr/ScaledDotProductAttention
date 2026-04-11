#pragma once

#include "tensor.hpp"

namespace attention::matmul {
    void matMulNaive(const Tensor &lhs, const Tensor &rhs, Tensor &result, std::size_t batch_idx);
    void matMulCacheOptimized(const Tensor &lhs, const Tensor &rhs, Tensor &result, std::size_t batch_idx);
    void matMulTiling(const Tensor &lhs, const Tensor &rhs, Tensor &result, std::size_t batch_idx);
}
