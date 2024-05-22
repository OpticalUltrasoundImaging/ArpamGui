cmake_minimum_required(VERSION 3.25)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)

project(ARPAM_gui
    LANGUAGES C CXX
)

enable_testing()

# Build libarpam
add_subdirectory(libuspam)

# Build the CLI
add_subdirectory(cli)

# Build the QT Gui
add_subdirectory(ArpamGuiQt)

