vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO kwsp/fftconv
    REF 977c696
    SHA512 beba638de7bf28dbc0e1ec9d1ad6a358fc54293165897eb59b83dc6b8702957862e99e48a01a8165754ac39dce4a73c235d625848fc97176a2b0450e2d528494
    HEAD_REF main
)

# Copy header only lib
file(INSTALL ${SOURCE_PATH}/fftconv_fftw/fftconv.hpp
    DESTINATION ${CURRENT_PACKAGES_DIR}/include
)

# Handle copyright
file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/fftconv" RENAME copyright)