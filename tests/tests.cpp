#include <gtest/gtest.h>

#include "matmul_unit_tests.hpp"
#include "tensor_unit_tests.hpp"


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}