set(EXE_NAME ARPAM_gui)
add_executable(${EXE_NAME}
    main.cpp
    main_gui.cpp
    vulkan_texture_loader.cpp
    utils/utils.cpp
)

# Use strict C++ 20
target_compile_options(${EXE_NAME} PRIVATE cxx_std_20)
set_target_properties(${EXE_NAME} PROPERTIES CXX_EXTENSIONS OFF)

# Enable link time optimization
set_target_properties(${EXE_NAME} PROPERTIES INTERPROCEDURAL_OPTIMIZATION TRUE)


target_link_libraries(${EXE_NAME} 
    PRIVATE 
    libarpam
    imgui
)