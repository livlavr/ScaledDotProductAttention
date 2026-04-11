add_executable(scaled_dot_product_attention_tests
    tests.cpp
)

target_link_libraries(scaled_dot_product_attention_tests
    PRIVATE
    scaled_dot_product_attention
    gtest_main
)

include(GoogleTest)
gtest_discover_tests(scaled_dot_product_attention_tests)
