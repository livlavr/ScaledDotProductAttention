include(FetchContent)
FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/heads/main.zip
)
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(unit_tests
        scaled_dot_product_attention/tests.cpp
)

target_link_libraries(unit_tests
        PRIVATE
        attention::core
        GTest::gtest_main
)

add_test(NAME AllUnitTests COMMAND unit_tests)