project(ArpamGuiQt VERSION 1.0.0 LANGUAGES CXX)

set(EXE_NAME ArpamGuiQt)

find_package(Vulkan REQUIRED)
find_package(Qt6 CONFIG REQUIRED COMPONENTS Widgets Gui)
qt_standard_project_setup()
set(CMAKE_AUTOMOC ON)

qt_add_executable(${EXE_NAME}
    src/main.cpp
    src/MainWindow.hpp
    src/MainWindow.cpp
    src/ImshowCanvas.hpp
    src/ImshowCanvas.cpp
    src/ImshowCanvasAnnotations.hpp
    src/ImshowCanvasAnnotations.cpp
    src/ImshowCanvasCursorState.hpp
    src/ImshowCanvasTicks.hpp
    src/ImshowCanvasTicks.cpp
    src/CoregDisplay.hpp
    src/CoregDisplay.cpp
    src/DataProcWorker.hpp
    src/DataProcWorker.cpp
    src/ReconParamsController.hpp
    src/ReconParamsController.cpp
    src/FrameController.hpp
    src/FrameController.cpp
    src/geometryUtils.cpp
    src/geometryUtils.hpp
)
qt_add_resources(${EXE_NAME} imageresources
    PREFIX "/"
    FILES 
    resources/images/radial_380.png
)
qt_add_resources(${EXE_NAME} styles
    PREFIX "/"
    FILES
    resources/styles/styles.qss
)

target_include_directories(${EXE_NAME} PRIVATE src)

target_link_libraries(${EXE_NAME}
    PRIVATE
    libuspam
    Qt::Widgets
    Qt::Gui
    Vulkan::Vulkan
)

set_target_properties(${EXE_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
    # INTERPROCEDURAL_OPTIMIZATION TRUE
)

if (WIN32)
    # On Windows, call windeployqt create a deployable folder
    # https://doc.qt.io/qt-6/windows-deployment.html
    set_target_properties(${EXE_NAME} PROPERTIES
        WIN32_EXECUTABLE ON
    )

    # Add post build command to call windeployqt
    add_custom_command(TARGET ${EXE_NAME} POST_BUILD
        COMMAND ${Qt6_DIR}/../../tools/Qt6/bin/windeployqt.exe 
        --verbose 1
        "$<IF:$<CONFIG:Debug>,--debug,--release>"
        "$<TARGET_FILE:${EXE_NAME}>"
    )

elseif (APPLE)
    # Post-build steps for macOS
    # https://doc.qt.io/qt-6/macos-deployment.html
    set_target_properties(${PROJECT_NAME} PROPERTIES
        MACOSX_BUNDLE ON
        MACOSX_BUNDLE_INFO_PLIST cmake/MacOSXBundleInfo.plist.in
    )

    # Use macdeployqt to bundle Qt libraries
    add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
        COMMAND ${Qt6_DIR}/../../tools/Qt6/bin/macdeployqt 
        "$<TARGET_BUNDLE_DIR:${EXE_NAME}>"
        -verbose=1
        -always-overwrite
        -codesign=-
        "$<IF:$<CONFIG:Release>,-dmg,-use-debug-libs>"
        COMMENT "Bundling Qt libraries"
    )

endif()
