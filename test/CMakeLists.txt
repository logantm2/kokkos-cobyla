add_executable(unit_tests "${PROJECT_SOURCE_DIR}/test/unit_tests.cpp")

target_link_libraries(unit_tests Kokkos::kokkos)
target_link_libraries(unit_tests GTest::gtest GTest::gtest_main pthread)

target_include_directories(unit_tests PUBLIC "${PROJECT_SOURCE_DIR}")

gtest_discover_tests(
    unit_tests
    TEST_PREFIX "kokkos-cobyla."
)
