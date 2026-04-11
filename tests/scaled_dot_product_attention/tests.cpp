#include <gtest/gtest.h>

#include "suites/matmul_unit_tests.hpp"
#include "suites/tensor_unit_tests.hpp"


int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
