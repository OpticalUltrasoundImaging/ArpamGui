cmake_minimum_required(VERSION 3.20)
project(libuspam LANGUAGES CXX CUDA)

find_package(Armadillo CONFIG REQUIRED)
find_package(OpenBLAS CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)

find_package(CUDA REQUIRED)

# Define libuspam library
set(LIB_NAME libuspam)
add_library(${LIB_NAME} STATIC
    src/signal.cpp
    src/recon.cpp
    src/cudaSignal.cu
)
target_include_directories(${LIB_NAME} PUBLIC include)

set_property(TARGET ${LIB_NAME} PROPERTY CUDA_ARCHITECTURES 75)
target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                       --use_fast_math
                       --extended-lambda
                       >)

target_link_libraries(${LIB_NAME}
    PUBLIC
    fftconv
    armadillo
    OpenBLAS::OpenBLAS
    opencv_world
)

target_include_directories(${LIB_NAME} PUBLIC ${CUDA_INCLUDE_DIRS})
target_link_libraries(${LIB_NAME} PUBLIC ${CUDA_LIBRARIES} ${CUDA_CUFFT_LIBRARIES})

# Use strict C++20
set_target_properties(${LIB_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
)

if(WIN32)
    target_compile_definitions(${LIB_NAME} PRIVATE _USE_MATH_DEFINES)
endif()

set_target_properties(${LIB_NAME} PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)


# Build tests
enable_testing()
find_package(GTest CONFIG REQUIRED)

add_executable(test_libuspam
    test/test_io.cpp
    test/test_signal.cpp
    test/test_recon.cpp
    test/test_cudaSignal.cu
 )

set_property(TARGET test_libuspam PROPERTY CUDA_ARCHITECTURES 75)

target_link_libraries(test_libuspam
    PRIVATE
    libuspam
    armadillo
    GTest::gtest
    GTest::gtest_main
)
include(GoogleTest)
gtest_discover_tests(test_libuspam)
