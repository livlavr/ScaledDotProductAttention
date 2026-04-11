#pragma once

#include <cstddef>
#include <vector>
#include <cassert>
#include <span>

#include "details.hpp"

namespace attention {
    class Tensor {
    public:
        Tensor(const std::size_t batches, const std::size_t rows, const std::size_t cols)
            : batches_count_(batches), rows_count_(rows), cols_count_(cols),
              data_(batches * rows * cols, 0.0f) {
        }

        Tensor(Tensor&&) noexcept = default;
        Tensor& operator=(Tensor&&) noexcept = default;

    private:
        Tensor(const Tensor& other) = default;

    public:
        [[nodiscard]] Tensor clone() const {
            return {*this};
        }

    public:
        float &operator()(const std::size_t batch_idx,
                          const std::size_t row_idx,
                          const std::size_t col_idx) noexcept {
            assert(batch_idx < batches() &&
                   row_idx < rows() &&
                   col_idx < cols() &&
                   details::kMsgIndexOutOfRange);

            return data_[get_index(batch_idx, row_idx, col_idx)];
        }

        const float &operator()(const std::size_t batch_idx,
                                const std::size_t row_idx,
                                const std::size_t col_idx) const noexcept {
            assert(batch_idx < batches() &&
                   row_idx < rows() &&
                   col_idx < cols() &&
                   details::kMsgIndexOutOfRange);

            return data_[get_index(batch_idx, row_idx, col_idx)];
        }

    public:
        [[nodiscard]] std::span<float> getBatchView(const std::size_t batch_idx) noexcept {
            assert(batch_idx < batches() && details::kMsgBatchIndexOutOfRange);
            return {data() + get_batch_offset(batch_idx), batch_stride()};
        }

        [[nodiscard]] std::span<const float> getBatchView(const std::size_t batch_idx) const noexcept {
            assert(batch_idx < batches() && details::kMsgBatchIndexOutOfRange);
            return {data() + get_batch_offset(batch_idx), batch_stride()};
        }

    public:
        [[nodiscard]] float *data() noexcept { return data_.data(); }
        [[nodiscard]] const float *data() const noexcept { return data_.data(); }

    public:
        static constexpr std::size_t rank() noexcept { return details::kDefaultRank; }

        [[nodiscard]] std::size_t batches() const noexcept { return batches_count_; }
        [[nodiscard]] std::size_t rows()    const noexcept { return rows_count_; }
        [[nodiscard]] std::size_t cols()    const noexcept { return cols_count_; }
        [[nodiscard]] std::size_t size()    const noexcept { return data_.size(); }

    public:
        [[nodiscard]] Tensor transposed() const {
            Tensor result(batches(), cols(), rows());

            for (std::size_t b_idx = 0; b_idx < batches(); ++b_idx) {
                for (std::size_t r_idx = 0; r_idx < rows(); ++r_idx) {
                    for (std::size_t c_idx = 0; c_idx < cols(); ++c_idx) {
                        result(b_idx, c_idx, r_idx) = (*this)(b_idx, r_idx, c_idx);
                    }
                }
            }
            return result;
        }

        void transpose() {
            if (rows_count_ == cols_count_) {
                for (std::size_t b_idx = 0; b_idx < batches_count_; ++b_idx) {
                    for (std::size_t r_idx = 0; r_idx < rows_count_; ++r_idx) {
                        for (std::size_t c_idx = r_idx + 1; c_idx < cols_count_; ++c_idx) {
                            std::swap((*this)(b_idx, r_idx, c_idx), (*this)(b_idx, c_idx, r_idx));
                        }
                    }
                }
            } else {
                *this = this->transposed();
            }
        }

    private:
        [[nodiscard]] std::size_t get_index(const std::size_t batch_idx,
                              const std::size_t row_idx,
                              const std::size_t col_idx) const noexcept {
            return batch_idx * batch_stride() + row_idx * cols() + col_idx;
        }

        [[nodiscard]] std::size_t get_batch_offset(const std::size_t batch_idx) const noexcept {
            return batch_idx * batch_stride();
        }

        [[nodiscard]] std::size_t batch_stride() const noexcept {
            return rows() * cols();
        }

    private:
        std::size_t batches_count_ = 0;
        std::size_t rows_count_    = 0;
        std::size_t cols_count_    = 0;

        std::vector<float> data_;
    };
}