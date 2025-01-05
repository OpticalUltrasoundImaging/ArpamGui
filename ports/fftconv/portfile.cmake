vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO kwsp/fftconv
    REF 271a9a156a35420bebba5c1a4cd1c6a257eb0de1  
    SHA512 13d617349754e8cbf020ff0165640e92b4c8c71b7036cbbde904c2bc2dba2475d49150a737347d1a5d0b73cc88eade76bd6a7c6100afc2e80b906d38115005dd
    HEAD_REF main
)

# Copy header only lib
file(INSTALL ${SOURCE_PATH}/include/fftconv DESTINATION ${CURRENT_PACKAGES_DIR}/include)

# Handle copyright
file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/fftconv" RENAME copyright)
