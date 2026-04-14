#pragma once

#include <gtest/gtest.h>
#include <vector>
#include <random>

#include "tensor.hpp"
#include "matrix_multiplication.hpp"

using namespace attention;
using namespace attention::matmul;

using MatMulFunc = void(*)(const Tensor&, const Tensor&, Tensor&, std::size_t);

class MatMulTest : public testing::TestWithParam<MatMulFunc> {
protected:
    static void expectTensorEqual(const Tensor& actual, const std::vector<float>& expected_data, const std::size_t batch_idx) {
        auto view = actual.getBatchView(batch_idx);
        ASSERT_EQ(view.size(), expected_data.size());

        for (std::size_t i = 0; i < view.size(); ++i) {
            EXPECT_NEAR(view[i], expected_data[i], 1e-1f);
        }
    }
};

TEST_P(MatMulTest, NormalSquareMatrices) {
    auto matMul = GetParam();

    Tensor A(1, 2, 2);
    Tensor B(1, 2, 2);
    Tensor C(1, 2, 2);

    A(0, 0, 0) = 1; A(0, 0, 1) = 2;
    A(0, 1, 0) = 3; A(0, 1, 1) = 4;

    B(0, 0, 0) = 5; B(0, 0, 1) = 6;
    B(0, 1, 0) = 7; B(0, 1, 1) = 8;

    matMul(A, B, C, 0);

    std::vector<float> expected = {19, 22, 43, 50};
    expectTensorEqual(C, expected, 0);
}

TEST_P(MatMulTest, NormalRectangularMatrices) {
    auto matMul = GetParam();

    Tensor A(1, 2, 3);
    Tensor B(1, 3, 1);
    Tensor C(1, 2, 1);

    A(0, 0, 0) = 1; A(0, 0, 1) = 2; A(0, 0, 2) = 3;
    A(0, 1, 0) = 4; A(0, 1, 1) = 5; A(0, 1, 2) = 6;

    B(0, 0, 0) = 7;
    B(0, 1, 0) = 8;
    B(0, 2, 0) = 9;

    matMul(A, B, C, 0);

    std::vector<float> expected = {50, 122};
    expectTensorEqual(C, expected, 0);
}

TEST_P(MatMulTest, TransposedRHSMatrix) {
    auto matMul = GetParam();

    Tensor Q(1, 2, 3);
    Tensor K(1, 2, 3);
    Tensor C(1, 2, 2);

    Q(0, 0, 0) = 1; Q(0, 0, 1) = 2; Q(0, 0, 2) = 3;
    Q(0, 1, 0) = 4; Q(0, 1, 1) = 5; Q(0, 1, 2) = 6;

    K(0, 0, 0) = 1; K(0, 0, 1) = 0; K(0, 0, 2) = 1;
    K(0, 1, 0) = 0; K(0, 1, 1) = 1; K(0, 1, 2) = 0;

    K.transpose();
    matMul(Q, K, C, 0);

    std::vector<float> expected = {4, 2, 10, 5};
    expectTensorEqual(C, expected, 0);
}

#ifndef NDEBUG
TEST_P(MatMulTest, DeathOnInvalidDimensions) {
    auto matMul = GetParam();

    Tensor A(1, 2, 3);
    Tensor B(1, 4, 2);
    Tensor C(1, 2, 2);

    EXPECT_DEATH({ matMul(A, B, C, 0); }, ".*");
}

TEST_P(MatMulTest, DeathOnInvalidOutputDimension) {
    auto matMul = GetParam();

    Tensor A(1, 2, 2);
    Tensor B(1, 2, 2);
    Tensor C(1, 3, 2);

    EXPECT_DEATH({ matMul(A, B, C, 0); }, ".*");
}
#endif

INSTANTIATE_TEST_SUITE_P(
    UniversalImplementations,
    MatMulTest,
    testing::Values(&matMulNaive, &matMulDirectNaive, &matMulCacheOptimized, &matMulSafe)
);

class MatMulAlignedTest : public testing::TestWithParam<MatMulFunc> {
protected:
    static void fillRandom(Tensor& t) {
        std::mt19937 gen(71);
        std::uniform_real_distribution<float> dist(-100, 100);

        for (std::size_t i = 0; i < t.size(); ++i) {
            t.data()[i] = dist(gen);
        }
    }

    static void expectTensorsEqual(const Tensor& actual, const Tensor& expected, const std::size_t batch_idx) {
        auto view_act = actual.getBatchView(batch_idx);
        auto view_exp = expected.getBatchView(batch_idx);

        ASSERT_EQ(view_act.size(), view_exp.size());

        for (std::size_t i = 0; i < view_act.size(); ++i) {
            EXPECT_NEAR(view_act[i], view_exp[i], 1e-1f);
        }
    }
};

TEST_P(MatMulAlignedTest, AlignedSquareMatrices) {
    auto matMul = GetParam();
    const std::size_t B = details::kBlockSize;

    Tensor lhs(1, B, B);
    Tensor rhs(1, B, B);
    Tensor result(1, B, B);
    Tensor expected(1, B, B);

    fillRandom(lhs);
    fillRandom(rhs);

    matMulNaive(lhs, rhs, expected, 0);

    matMul(lhs, rhs, result, 0);

    expectTensorsEqual(result, expected, 0);
}

TEST_P(MatMulAlignedTest, AlignedRectangularMatrices) {
    auto matMul = GetParam();
    const std::size_t B = details::kBlockSize;

    Tensor lhs(1, 2 * B, B);
    Tensor rhs(1, B, 3 * B);
    Tensor result(1, 2 * B, 3 * B);
    Tensor expected(1, 2 * B, 3 * B);

    fillRandom(lhs);
    fillRandom(rhs);

    matMulNaive(lhs, rhs, expected, 0);
    matMul(lhs, rhs, result, 0);

    expectTensorsEqual(result, expected, 0);
}

INSTANTIATE_TEST_SUITE_P(
    AlignedImplementations,
    MatMulAlignedTest,
    testing::Values(&matMulTiling, &matMulSIMD)
);