find_package(Qt6 REQUIRED COMPONENTS Widgets PrintSupport)

add_library(qcustomplot STATIC qcustomplot.cpp)
target_include_directories(qcustomplot PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(qcustomplot PUBLIC Qt6::Widgets Qt6::PrintSupport)
set_target_properties(qcustomplot PROPERTIES AUTOMOC ON CXX_STANDARD 17)
