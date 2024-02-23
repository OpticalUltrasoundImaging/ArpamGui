cmake_minimum_required(VERSION 3.20)
project(libarpam C CXX)

set(CMAKE_CXX_STANDARD 20)

find_package(Eigen3 CONFIG REQUIRED)

add_library(libarpam STATIC)
target_sources(libarpam
    PRIVATE
    src/libarpam/libarpam.cpp
)
target_include_directories(libarpam PUBLIC src)
target_link_libraries(libarpam 
    PUBLIC
    fftconv
    Eigen3::Eigen
)


enable_testing()
find_package(GTest CONFIG REQUIRED)

add_executable(test_libarpam)
target_sources(test_libarpam
    PRIVATE
    test/test_libarpam.cpp
    test/test_io.cpp
)
target_link_libraries(test_libarpam
    PRIVATE
    libarpam
    Eigen3::Eigen
    GTest::gtest
    GTest::gtest_main
)
include(GoogleTest)
gtest_discover_tests(test_libarpam)