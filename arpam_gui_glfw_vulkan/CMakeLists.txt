set(EXE_NAME ARPAM_gui)
add_executable(${EXE_NAME}
    main.cpp
    main_gui.cpp
    vulkan_texture_loader.cpp
    utils/utils.cpp
)

# Use strict C++ 20
set_target_properties(${EXE_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
    INTERPROCEDURAL_OPTIMIZATION TRUE  # LTO
)

target_link_libraries(${EXE_NAME} 
    PRIVATE 
    libarpam
    imgui
)
