include(FetchContent)

FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/heads/main.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(unit_tests tests.cpp)

target_include_directories(unit_tests PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/suites
)

target_link_libraries(unit_tests
        PRIVATE
        attention::core
        gtest_main
)

include(GoogleTest)
gtest_discover_tests(unit_tests)
