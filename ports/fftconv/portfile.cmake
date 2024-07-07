vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO kwsp/fftconv
    REF b6af7f2
    SHA512 04f2a9e324514f452c232a893babe4116e810b301cb33ddb2eb4732b3f62eb96a34c3f367a219617eabffe7783f2cda469ed2290c4fa87dd8efd7257ce053fd3
    HEAD_REF main
)

# Copy header only lib
file(INSTALL ${SOURCE_PATH}/fftconv_fftw/fftconv.hpp
    DESTINATION ${CURRENT_PACKAGES_DIR}/include
)

# Handle copyright
file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/fftconv" RENAME copyright)