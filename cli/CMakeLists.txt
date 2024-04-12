find_package(CLI11 CONFIG REQUIRED)
find_package(indicators CONFIG REQUIRED)

add_executable(arpam
    main.cpp
)
set_target_properties(arpam PROPERTIES
    CXX_STANDARD 20
    CXX_EXTENSIONS OFF
    # INTERPROCEDURAL_OPTIMIZATION TRUE  # LTO
)
target_link_libraries(arpam
    PRIVATE
    libuspam
    CLI11::CLI11
    indicators::indicators
)
