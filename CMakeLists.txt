cmake_minimum_required(VERSION 3.25)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)

# Enable CCache if available
find_program(CCACHE_PROGRAM ccache)
if(CCACHE_PROGRAM)
  message(STATUS "Found CCache: ${CCACHE_PROGRAM}")
  set(CMAKE_C_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
  set(CMAKE_CXX_COMPILER_LAUNCHER ${CCACHE_PROGRAM})
else()
  message(STATUS "CCache not found")
endif()

project(ARPAM_gui
    LANGUAGES C CXX
)

enable_testing()

# Build libarpam
add_subdirectory(libuspam)

# Build the QT Gui
add_subdirectory(ArpamGuiQt)

