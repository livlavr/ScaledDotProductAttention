#pragma once

#include <gtest/gtest.h>
#include <utility>
#include <numeric>

#include "tensor.hpp"

using namespace attention;

TEST(TensorTest, BasicDimensions) {
    Tensor t(2, 10, 8);
    EXPECT_EQ(t.batches(), 2);
    EXPECT_EQ(t.rows(), 10);
    EXPECT_EQ(t.cols(), 8);
    EXPECT_EQ(t.size(), 2 * 10 * 8);
}

TEST(TensorTest, ZeroInitialization) {
    Tensor t(2, 3, 4);
    for (size_t i = 0; i < t.size(); ++i) {
        EXPECT_FLOAT_EQ(t.data()[i], 0.0f);
    }
}

TEST(TensorTest, DataAccess) {
    Tensor t(1, 1, 1);
    t(0, 0, 0) = 3.14f;
    EXPECT_FLOAT_EQ(t(0, 0, 0), 3.14f);
}

TEST(TensorTest, LinearMemoryLayout) {
    Tensor t(2, 3, 4);
    std::iota(t.data(), t.data() + t.size(), 0.0f);

    EXPECT_FLOAT_EQ(t(0, 0, 0), 0.0f);
    EXPECT_FLOAT_EQ(t(0, 1, 2), 1.0f * 4 + 2.0f);
    EXPECT_FLOAT_EQ(t(1, 0, 0), 1.0f * 3 * 4);
    EXPECT_FLOAT_EQ(t(1, 2, 3), 1.0f * 12 + 2.0f * 4 + 3.0f);
}

TEST(TensorTest, ConstMethods) {
    Tensor t(2, 3, 4);
    t(1, 2, 3) = 42.0f;

    const Tensor& ct = t;
    EXPECT_FLOAT_EQ(ct(1, 2, 3), 42.0f);

    auto view = ct.getBatchView(1);
    EXPECT_EQ(view.size(), 3 * 4);
    EXPECT_FLOAT_EQ(view[2 * 4 + 3], 42.0f);
}

TEST(TensorTest, TransposeDimensions) {
    Tensor t(2, 10, 8);
    t.transpose();

    EXPECT_EQ(t.batches(), 2);
    EXPECT_EQ(t.rows(), 8);
    EXPECT_EQ(t.cols(), 10);
}

TEST(TensorTest, TransposeData) {
    Tensor t(2, 3, 4);
    float val = 1.0f;
    for (size_t b = 0; b < t.batches(); ++b) {
        for (size_t s = 0; s < t.rows(); ++s) {
            for (size_t d = 0; d < t.cols(); ++d) {
                t(b, s, d) = val++;
            }
        }
    }

    Tensor t_trans = t.transposed();

    for (size_t b = 0; b < t.batches(); ++b) {
        for (size_t s = 0; s < t.rows(); ++s) {
            for (size_t d = 0; d < t.cols(); ++d) {
                EXPECT_FLOAT_EQ(t_trans(b, d, s), t(b, s, d));
            }
        }
    }
}

TEST(TensorTest, BatchView) {
    Tensor t(2, 3, 4);
    t(1, 0, 0) = 99.0f;

    auto view = t.getBatchView(1);

    EXPECT_EQ(view.size(), 3 * 4);
    EXPECT_EQ(view.data(), t.data() + (3 * 4));
    EXPECT_FLOAT_EQ(view[0], 99.0f);

    view[1] = 77.0f;
    EXPECT_FLOAT_EQ(t(1, 0, 1), 77.0f);
}

TEST(TensorTest, MoveConstructor) {
    Tensor t1(2, 4, 4);
    float* raw_ptr = t1.data();

    Tensor t2 = std::move(t1);

    EXPECT_EQ(t2.batches(), 2);
    EXPECT_EQ(t2.data(), raw_ptr);
}

TEST(TensorTest, MoveAssignment) {
    Tensor t1(2, 4, 4);
    float* raw_ptr = t1.data();

    Tensor t2(1, 1, 1);
    t2 = std::move(t1);

    EXPECT_EQ(t2.batches(), 2);
    EXPECT_EQ(t2.rows(), 4);
    EXPECT_EQ(t2.data(), raw_ptr);
}

#ifndef NDEBUG
TEST(TensorTest, OutOfRangeAssertions) {
    Tensor t(2, 3, 4);

    EXPECT_DEATH(t(2, 0, 0), ".*");

    EXPECT_DEATH(t(0, 3, 0), ".*");

    EXPECT_DEATH(t(0, 0, 4), ".*");

    EXPECT_DEATH(static_cast<void>(t.getBatchView(2)), ".*");
}
#endif
