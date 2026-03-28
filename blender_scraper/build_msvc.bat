@echo off
REM Build blender_scraper with MSVC on Windows
REM Requires: CMake, Visual Studio, vcpkg with curl installed

SET BUILD_TYPE=Release
SET BUILD_DIR=build\%BUILD_TYPE%

echo === Building blender_scraper [%BUILD_TYPE%] ===

IF NOT DEFINED VCPKG_ROOT (
    echo WARNING: VCPKG_ROOT not set. Set it if cmake can't find libcurl.
    echo Example: set VCPKG_ROOT=C:\vcpkg
)

cmake -S . -B %BUILD_DIR% ^
    -DCMAKE_BUILD_TYPE=%BUILD_TYPE% ^
    -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake"

cmake --build %BUILD_DIR% --config %BUILD_TYPE%

echo.
echo === Build complete ===
echo Binary: %BUILD_DIR%\%BUILD_TYPE%\blender_scraper.exe
echo.
echo Quick start:
echo   copy config.example.json config.json
echo   REM Edit config.json and set your github_token
echo   %BUILD_DIR%\%BUILD_TYPE%\blender_scraper.exe --help
