set(VCPKG_TARGET_ARCHITECTURE arm64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE static)

set(VCPKG_CMAKE_SYSTEM_NAME Darwin)
set(VCPKG_OSX_ARCHITECTURES arm64)

### custom
# Disable compiler tracking to increase binary cache hit rate
# https://learn.microsoft.com/en-us/vcpkg/users/triplets
# https://github.com/microsoft/vcpkg/pull/17526
set(VCPKG_DISABLE_COMPILER_TRACKING ON)
