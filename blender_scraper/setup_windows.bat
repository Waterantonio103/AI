@echo off
REM Full Windows setup: installs vcpkg, curl, then builds blender_scraper
REM Run this from a Developer Command Prompt OR it will attempt to find VS automatically.
REM Requires: Git, CMake, Visual Studio 2019+ (any edition including Community)

setlocal enabledelayedexpansion

echo ============================================
echo  blender_scraper Windows Setup
echo ============================================
echo.

REM --- Check prerequisites ---
where git >nul 2>&1
IF ERRORLEVEL 1 (
    echo [ERROR] git not found.
    echo Download from: https://git-scm.com/download/win
    exit /b 1
)

where cmake >nul 2>&1
IF ERRORLEVEL 1 (
    REM winget installs CMake here — add it to PATH for this session
    SET "CMAKE_DEFAULT=C:\Program Files\CMake\bin"
    IF EXIST "%CMAKE_DEFAULT%\cmake.exe" (
        SET "PATH=%CMAKE_DEFAULT%;%PATH%"
        echo [info] Added CMake to PATH: %CMAKE_DEFAULT%
    ) ELSE (
        echo [ERROR] cmake not found. Restart your terminal after installing,
        echo         or add CMake\bin to your PATH manually.
        exit /b 1
    )
)

echo [OK] git and cmake found.
echo.

REM --- Install vcpkg into C:\vcpkg ---
SET VCPKG_ROOT=C:\vcpkg

IF NOT EXIST "%VCPKG_ROOT%\vcpkg.exe" (
    echo [1/4] Cloning vcpkg into %VCPKG_ROOT% ...
    git clone https://github.com/microsoft/vcpkg.git "%VCPKG_ROOT%"
    IF ERRORLEVEL 1 ( echo [ERROR] git clone failed. & exit /b 1 )

    echo [2/4] Bootstrapping vcpkg ...
    call "%VCPKG_ROOT%\bootstrap-vcpkg.bat" -disableMetrics
    IF ERRORLEVEL 1 ( echo [ERROR] bootstrap failed. & exit /b 1 )
) ELSE (
    echo [1/4] vcpkg already installed at %VCPKG_ROOT%
    echo [2/4] Skipping bootstrap.
)

echo.
echo [3/4] Installing libcurl (x64-windows) via vcpkg ...
"%VCPKG_ROOT%\vcpkg.exe" install curl:x64-windows nlohmann-json:x64-windows
IF ERRORLEVEL 1 ( echo [ERROR] vcpkg install failed. & exit /b 1 )

echo.
echo [4/4] Building blender_scraper ...
SET BUILD_DIR=build\Release

cmake -S . -B "%BUILD_DIR%" ^
    -DCMAKE_BUILD_TYPE=Release ^
    -DCMAKE_TOOLCHAIN_FILE="%VCPKG_ROOT%\scripts\buildsystems\vcpkg.cmake"
IF ERRORLEVEL 1 ( echo [ERROR] cmake configure failed. & exit /b 1 )

cmake --build "%BUILD_DIR%" --config Release
IF ERRORLEVEL 1 ( echo [ERROR] cmake build failed. & exit /b 1 )

echo.
echo ============================================
echo  Build complete!
echo  Binary: %BUILD_DIR%\Release\blender_scraper.exe
echo ============================================
echo.
echo Next steps:
echo   1. copy config.example.json config.json
echo   2. Edit config.json and paste your GitHub token
echo      (get one at: https://github.com/settings/tokens)
echo   3. Run: %BUILD_DIR%\Release\blender_scraper.exe
echo.

REM Persist VCPKG_ROOT for future sessions
setx VCPKG_ROOT "%VCPKG_ROOT%" >nul
echo [info] VCPKG_ROOT=%VCPKG_ROOT% saved to user environment.
