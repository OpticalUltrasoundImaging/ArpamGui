cmake_minimum_required(VERSION 3.20)
project(libarpam C CXX)

find_package(Eigen3 CONFIG REQUIRED)

# Define libarpam library
add_library(libarpam STATIC)
target_sources(libarpam
    PRIVATE
    src/signal.cpp
    src/recon.cpp
)
target_include_directories(libarpam PUBLIC include)
target_link_libraries(libarpam
    PUBLIC
    fftconv
    Eigen3::Eigen
)

# Use strict C++20
set_target_properties(libarpam PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
)

if(WIN32)
    target_compile_definitions(libarpam PRIVATE _USE_MATH_DEFINES)
endif()


# Build tests
enable_testing()
find_package(GTest CONFIG REQUIRED)

add_executable(test_libarpam)
target_sources(test_libarpam
    PRIVATE
    test/test_libarpam.cpp
    test/test_io.cpp
    test/test_signal.cpp
    test/test_recon.cpp
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
