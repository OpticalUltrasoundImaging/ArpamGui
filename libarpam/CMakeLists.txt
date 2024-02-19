cmake_minimum_required(VERSION 3.20)
project(libarpam C CXX)

set(CMAKE_CXX_STANDARD 20)

add_library(libarpam STATIC)
target_sources(libarpam 
    PRIVATE
    libarpam.cpp
)
target_include_directories(libarpam PUBLIC .)

target_link_libraries(libarpam PUBLIC fftconv)