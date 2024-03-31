### Build ImGui and backends (GLFW, Vulkan)

# GLFW
find_package(glfw3 CONFIG REQUIRED)

# Vulkan
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -DVK_PROTOTYPES")
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DVK_PROTOTYPES")
find_package(Vulkan REQUIRED)

# Compile ImGui and ImPlot into a single static library
add_library(imgui STATIC)

set(IMGUI_DIR ./extern/imgui)
set(IMGUI_SRC
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp)

set(IMPLOT_DIR ./extern/implot)
set(IMPLOT_SRC
    ${IMPLOT_DIR}/implot.cpp
    ${IMPLOT_DIR}/implot_items.cpp
    ${IMPLOT_DIR}/implot_demo.cpp
)

target_sources(imgui
    PRIVATE
    ${IMGUI_SRC}
    ${IMPLOT_SRC}
)
target_include_directories(imgui
    PUBLIC 
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
    ${IMPLOT_DIR}
)
target_link_libraries(imgui
    PUBLIC
    "glfw"
    "Vulkan::Vulkan"
)