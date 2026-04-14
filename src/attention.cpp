#include <cmath>
#include <limits>
#include <algorithm>

#include "tensor.hpp"
#include "matrix_multiplication.hpp"
#include "attention.hpp"

namespace attention {
    using namespace matmul;

    namespace details {
        inline void scale(std::span<float> row, const float factor) {
            for (float &val: row) {
                val *= factor;
            }
        }

        inline void mask(std::span<float> row, const std::size_t row_idx) {
            for (std::size_t j = row_idx + 1; j < row.size(); ++j) {
                row[j] = -std::numeric_limits<float>::infinity();
            }
        }

        inline void softmax(std::span<float> row) {
            const float max_val = std::ranges::max(row);

            assert(std::isfinite(max_val) && "Softmax : Input contains NaN or Inf");

            float sum_exp = 0;
            for (float &val: row) {
                val = std::exp(val - max_val);
                sum_exp += val;
            }

            for (float &val: row) {
                val /= sum_exp;
            }
        }
    }

    Tensor attentionWithMatmul(
        const Tensor &Q,
        const Tensor &K,
        const Tensor &V,
        MatMulType matmul_type
    ) {
        Tensor result(Q.batches(), Q.rows(), V.cols());
        Tensor scores(Q.batches(), Q.rows(), K.rows());

        const float scale_factor = 1 / std::sqrt(static_cast<float>(Q.cols()));

        for (std::size_t batch = 0; batch < Q.batches(); ++batch) {
            matMul(matmul_type, Q, K, scores, batch);

            auto scores_batch = scores.getBatchView(batch);
            const std::size_t seq_k = K.rows();

            for (std::size_t i = 0; i < Q.rows(); ++i) {
                const auto row = scores_batch.subspan(i * seq_k, seq_k);

                details::scale(row, scale_factor);
                details::mask(row, i);
                details::softmax(row);
            }

            matMul(matmul_type, scores, V, result, batch);
        }

        return result;
    }
} // namespace attention
