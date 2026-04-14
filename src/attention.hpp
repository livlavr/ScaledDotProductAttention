#pragma once

#include "tensor.hpp"
#include "matrix_multiplication.hpp"

namespace attention {
    Tensor attentionWithMatmul(
        const Tensor &Q,
        const Tensor &K,
        const Tensor &V,
        matmul::MatMulType matmul_type
    );
} // namespace attention
