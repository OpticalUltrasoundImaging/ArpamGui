# ArpamGui

Control, data acquisition, real-time display, image processing, parameter optimization, and image annotation, all-in-one GUI for an acoustic resolution photoacoustic microscopy (ARPAM) coregistered with ultrasound endoscopy system in the Optical and Ultrasound Imaging Lab.

Data acquisition: AlazarTech SDK
GUI framework: Qt

## Building

Refer to the [CI script](./.github/workflows/build.yml) for the build process on Windows and macOS. This project is based on [kwsp/CppTemplate](https://github.com/kwsp/CppTemplate), which uses CMake + Ninja as the build system, VCPKG as the package manager, and Clang as the preferred compiler, with build presets for x64 Windows and arm64 macOS defined in CMakePresets.json. ArpamGui is currently only tested on x64 Windows and arm64 macOS but will probably work on other platforms.

### 1. Install a compiler and build system

Install Visual Studio, and in the installer, select "Desktop development with C++", and in individual components, make sure the latest MSVC compiler, CMake, Ninja, and optionally Clang is selected. Both CL and Clang work. Ninja builds more than twice as fast as MSBuild so it is preferred.

#### macOS

Install Xcode or command line tools for the Apple Clang compiler. Last I checked, Qt didn't build with Homebrew Clang.

Install the [homebrew package manager](https://brew.sh/), then run the following command to install mono (a C# runtime required by VCPKG's NuGet integration), ninja (build system), llvm (for clang-tidy), and some other tools required to build the dependencies.

```sh
brew install mono ninja llvm
ln -s $(brew --prefix llvm)/bin/clang-tidy /usr/local/bin/clang-tidy
brew install autoconf autoconf-archive automake libtool
```

### 2. Install the VCPKG package manager

Follow the instructions here: <https://github.com/microsoft/vcpkg>. On macOS, clone the repo to `~/vcpkg`, and on Windows, clone the repo to `C:/vcpkg`. On Windows, optionally use the `vcpkg` from the Visual Studio optional component.

### 3. Configure the project

In CMake lingo this means CMake build's the build system (Ninja); with VCPKG integration, this step takes ~30min on the first run because it also builds the dependencies defined in `vcpkg.json`. On subsequent runs, the prebuilt binaries of the dependencies are cached.

#### Windows

Activate the Visual Studio Developer PowerShell and run all commands inside the Developer PowerShell

3 CMake presets are provided for Windows

1. `clang-cl`: (preferred) x64 ClangCL + Ninja Multi-config
2. `cl`: x64 CL + Ninja Multi-config
3. `win64`: x64 CL + MSBuild

```powershell
cmake --preset clang-cl
```

#### macOS

`clang` is defined for Clang + Ninja Multi-Config

```sh
cmake --preset clang
```

### 4. Build the project

CMake build presets are defined with the `-debug`, `-relwithdebinfo`, and `-release` for "debug", "release with debug info", and "release" builds. For development, prefer `relwithdebinfo` unless something can't be debugged in this build.

#### Windows

```powershell
cmake --build --preset clang-cl-relwithdebinfo
```

#### macOS

```sh
cmake --build --preset clang-relwithdebinfo
```
