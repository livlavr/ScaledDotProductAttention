#pragma once

#include <gtest/gtest.h>
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
            EXPECT_NEAR(view[i], expected_data[i], 1e-5f);
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

TEST_P(MatMulTest, BatchIsolation) {
    auto matMul = GetParam();

    Tensor A(2, 2, 2);
    Tensor B(2, 2, 2);
    Tensor C(2, 2, 2);

    A(1, 0, 0) = 1; A(1, 0, 1) = 1; A(1, 1, 0) = 1; A(1, 1, 1) = 1;
    B(1, 0, 0) = 2; B(1, 0, 1) = 2; B(1, 1, 0) = 2; B(1, 1, 1) = 2;

    matMul(A, B, C, 1);

    std::vector<float> expected_batch_1 = {4, 4, 4, 4};
    expectTensorEqual(C, expected_batch_1, 1);

    std::vector<float> expected_batch_0 = {0, 0, 0, 0};
    expectTensorEqual(C, expected_batch_0, 0);
}

#ifndef NDEBUG
TEST_P(MatMulTest, DeathOnInvalidDimensions) {
    auto matMul = GetParam();

    Tensor A(1, 2, 3);
    Tensor B(1, 4, 2);
    Tensor C(1, 2, 2);

    EXPECT_DEATH({
        matMul(A, B, C, 0);
    }, ".*");
}

TEST_P(MatMulTest, DeathOnInvalidOutputDimension) {
    auto matMul = GetParam();

    Tensor A(1, 2, 2);
    Tensor B(1, 2, 2);
    Tensor C(1, 3, 2);

    EXPECT_DEATH({
        matMul(A, B, C, 0);
    }, ".*");
}
#endif

INSTANTIATE_TEST_SUITE_P(
    AllImplementations,
    MatMulTest,
    testing::Values(&matMulNaive, &matMulCacheOptimized, &matMulTiling)
);