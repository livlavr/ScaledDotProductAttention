#pragma once

#include "tensor.hpp"

namespace attention {
    namespace matmul { enum class MatMulType; }

    Tensor attentionWithMatmul(
        const Tensor &Q,
        const Tensor &K,
        const Tensor &V,
        matmul::MatMulType matmul_type
    );
} // namespace attention
