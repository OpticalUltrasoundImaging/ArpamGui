vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO kwsp/fftconv
    REF a6f2e67b6ca1d95668f7f56fe218807ee32ef4c5 
    SHA512 b20cd884da95250f355a7f9c0c3f5577a522433aa14f2a338fcc2966a33d1e9838e172c34da5518bfde209ac17f9edd2021b81fdde9f990a47dbc5ac1bd944a2
    HEAD_REF main
)

# Copy header only lib
file(INSTALL ${SOURCE_PATH}/include/fftconv DESTINATION ${CURRENT_PACKAGES_DIR}/include)

# Handle copyright
file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/fftconv" RENAME copyright)
