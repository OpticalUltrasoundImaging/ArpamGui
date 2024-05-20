project(arpam-gui-qt VERSION 1.0.0 LANGUAGES CXX)

set(EXE_NAME arpam-gui-qt)

find_package(Qt6 CONFIG REQUIRED COMPONENTS Widgets)
qt_standard_project_setup()
set(CMAKE_AUTOMOC ON)

qt_add_executable(${EXE_NAME}
    src/main.cpp
    src/MainWindow.hpp
    src/MainWindow.cpp
    src/ImshowCanvas.hpp
    src/ImshowCanvas.cpp
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
)

set_target_properties(${EXE_NAME} PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
    # INTERPROCEDURAL_OPTIMIZATION TRUE
    WIN32_EXECUTABLE ON
    MACOSX_BUNDLE ON
)

if (WIN32)
    # On Windows, call windeployqt create a deployable folder
    # https://doc.qt.io/qt-6/windows-deployment.html

    # Find windeployqt
    get_target_property(_qt_bin_dir Qt::qmake IMPORTED_LOCATION)
    get_filename_component(_qt_bin_dir "${_qt_bin_dir}" DIRECTORY)
    find_program(WINDEPLOYQT_EXE NAMES windeployqt.exe HINTS "${_qt_bin_dir}")
    find_program(WINDEPLOYQT_DEBUG_EXE NAMES windeployqt.debug.bat HINTS "${_qt_bin_dir}")

    message("CMAKE_CURRENT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}")
    message("_qt_bin_dir ${_qt_bin_dir}")
    message("WINDEPLOYQT_EXE ${WINDEPLOYQT_EXE}")
    message("WINDEPLOYQT_DEBUG_EXE ${WINDEPLOYQT_DEBUG_EXE}")

    # Add post build command to call windeployqt
    add_custom_command(TARGET ${EXE_NAME} POST_BUILD
        COMMAND "$<IF:$<CONFIG:Debug>,${WINDEPLOYQT_DEBUG_EXE},${WINDEPLOYQT_EXE}>"
        --verbose 0 
        --no-compiler-runtime 
        "$<TARGET_FILE:${EXE_NAME}>"
    )


    # On Windows, also copy the system runtime libraries
    include(InstallRequiredSystemLibraries)
    foreach(_file ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS})
        add_custom_command(TARGET ${EXE_NAME} POST_BUILD
                           COMMAND ${CMAKE_COMMAND} -E copy_if_different
                           ${_file} $<TARGET_FILE_DIR:${EXE_NAME}>)
    endforeach()
endif()
