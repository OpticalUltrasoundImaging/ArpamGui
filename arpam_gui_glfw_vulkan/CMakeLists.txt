add_executable(ARPAM_gui
    main.cpp
    main_gui.cpp
    vulkan_texture_loader.cpp
    utils/utils.cpp
)

target_link_libraries(ARPAM_gui 
    PRIVATE 
    libarpam
    imgui
)