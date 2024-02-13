cmake_minimum_required(VERSION 3.20)
project(ARPAM_gui C CXX)

set(CMAKE_CXX_STANDARD 20)

# VENDOR_LIBS is a list that stores all external dependencies of 
# ARPAM_gui, including imgui, glfw, vulkan
set(VENDOR_LIBS)

# Add ImGui and gui backends
include(CMakeLists_imgui.cmake)
message("Vendor Libraries: ${VENDOR_LIBS}")

# # Build an ImGui example
# add_executable(example_glfw_vulkan)
# set(EXAMPLE_DIR ./imgui/examples/example_glfw_vulkan)
# target_sources(example_glfw_vulkan
#     PRIVATE
#     ${EXAMPLE_DIR}/main.cpp)
# target_link_libraries(example_glfw_vulkan PRIVATE ${VENDOR_LIBS})
# target_compile_definitions(example_glfw_vulkan PUBLIC -DImTextureID=ImU64)

# Build the ARPAM gui
add_subdirectory(arpam_gui_glfw_vulkan)