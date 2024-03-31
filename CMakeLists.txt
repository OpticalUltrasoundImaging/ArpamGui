cmake_minimum_required(VERSION 3.25)
project(ARPAM_gui
    LANGUAGES C CXX
)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)

enable_testing()

### Add ImGui and gui backends
# Disable clang-tidy temporarily to build dependencies and restore after
set(CMAKE_CXX_CLANG_TIDY_SAVE ${CMAKE_CXX_CLANG_TIDY})
set(CMAKE_CXX_CLANG_TIDY "")
include(CMakeLists_imgui.cmake)
set(CMAKE_CXX_CLANG_TIDY ${CMAKE_CXX_CLANG_TIDY_SAVE})

# # Build an ImGui example
# add_executable(example_glfw_vulkan)
# set(EXAMPLE_DIR ./imgui/examples/example_glfw_vulkan)
# target_sources(example_glfw_vulkan
#     PRIVATE
#     ${EXAMPLE_DIR}/main.cpp)
# target_link_libraries(example_glfw_vulkan PRIVATE ${VENDOR_LIBS})
# target_compile_definitions(example_glfw_vulkan PUBLIC -DImTextureID=ImU64)

# Add fftconv
add_subdirectory(extern/fftconv/fftconv_fftw)

# Build libarpam
add_subdirectory(libarpam)

# Build the ARPAM gui
add_subdirectory(arpam_gui_glfw_vulkan)
