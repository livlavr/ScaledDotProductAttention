add_executable(unit_tests tests.cpp)

target_link_libraries(unit_tests
        PRIVATE
        attention::core
        gtest_main
)

include(GoogleTest)
gtest_discover_tests(unit_tests)