cmake_minimum_required(VERSION 3.20)
project(libuspam LANGUAGES CXX)

find_package(Armadillo CONFIG REQUIRED)
find_package(OpenBLAS CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(RapidJSON CONFIG REQUIRED)

# Define libuspam library
set(LIB_NAME libuspam)
add_library(${LIB_NAME} STATIC
    src/signal.cpp
    src/recon.cpp
    src/imutil.cpp
)
target_include_directories(${LIB_NAME} PUBLIC include)
target_link_libraries(${LIB_NAME}
    PUBLIC
    fftconv
    armadillo
    OpenBLAS::OpenBLAS
    opencv_world
    rapidjson
)

# Use strict C++20
set_target_properties(${LIB_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
)

if(WIN32)
    target_compile_definitions(${LIB_NAME} PRIVATE _USE_MATH_DEFINES)
endif()

# Optionally find CUDA
include(CheckLanguage)
check_language(CUDA)

message("check_language(CUDA): CMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}")

if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)

    set_target_properties(${LIB_NAME} PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES 75
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )

    target_compile_definitions(${LIB_NAME} PUBLIC USPAM_HAS_CUDA=1)
    target_sources(${LIB_NAME}
        PRIVATE
        src/cudaSignal.cu
        src/cudaRecon.cu
    )

    target_compile_options(${LIB_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                           --use_fast_math
                           --extended-lambda
                           >)

    message("BEGIN CUDA::cudart")
    get_target_property(CUDART_INCLUDE_DIRS CUDA::cudart INTERFACE_INCLUDE_DIRECTORIES)
    get_target_property(CUDART_LINK_LIBRARIES CUDA::cudart INTERFACE_LINK_LIBRARIES)
    message("CUDA::cudart include directories: ${CUDART_INCLUDE_DIRS}")
    message("CUDA::cudart link libraries: ${CUDART_LINK_LIBRARIES}")
    message("END CUDA::cudart")

    target_link_libraries(${LIB_NAME} PUBLIC CUDA::cudart CUDA::cublas CUDA::cufft)
else()
    message(STATUS "No CUDA support")
endif()


# Build tests
enable_testing()
find_package(GTest CONFIG REQUIRED)

add_executable(test_libuspam
    test/test_io.cpp
    test/test_signal.cpp
    test/test_recon.cpp
    test/test_imutil.cpp
)

if (CMAKE_CUDA_COMPILER)
    target_sources(test_libuspam
        PRIVATE
        test/test_cudaSignal.cu
        test/test_cudaRecon.cu
    )
    target_compile_options(test_libuspam PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                           --use_fast_math
                           --extended-lambda
                           >)
    set_target_properties(${LIB_NAME} PROPERTIES
        CUDA_STANDARD 20
        CUDA_STANDARD_REQUIRED ON
        CUDA_ARCHITECTURES 75
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_RESOLVE_DEVICE_SYMBOLS ON
    )
endif()


target_link_libraries(test_libuspam
    PRIVATE
    libuspam
    armadillo
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_libuspam)
