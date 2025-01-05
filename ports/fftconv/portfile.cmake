vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO kwsp/fftconv
    REF v0.5.1  
    SHA512 e89910050cc8cac2381ae6dd13c7c1f0e640065f9f4aefa329f44cadb6628534bee15ffa76cca04592fb39e46160454d5bf984e48d0bfef8e7d6b86d31971541
    HEAD_REF main
)

# Copy header only lib
file(INSTALL ${SOURCE_PATH}/include/fftconv DESTINATION ${CURRENT_PACKAGES_DIR}/include)

# Handle copyright
file(INSTALL "${SOURCE_PATH}/LICENSE" DESTINATION "${CURRENT_PACKAGES_DIR}/share/fftconv" RENAME copyright)
