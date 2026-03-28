#!/usr/bin/env bash
# Build the blender_scraper project (Linux/macOS/MSYS2/WSL)
set -e

BUILD_TYPE=${1:-Release}
BUILD_DIR="build/${BUILD_TYPE}"

echo "=== Building blender_scraper [${BUILD_TYPE}] ==="

# Dependencies check
if ! command -v cmake &>/dev/null; then
    echo "ERROR: cmake not found. Install it first."
    exit 1
fi
if ! command -v curl-config &>/dev/null && ! pkg-config --exists libcurl 2>/dev/null; then
    echo "WARNING: libcurl may not be found. Install libcurl-dev / curl-devel."
fi

mkdir -p "${BUILD_DIR}"
cmake -S . -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DCMAKE_EXPORT_COMPILE_COMMANDS=ON

cmake --build "${BUILD_DIR}" --parallel "$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)"

echo ""
echo "=== Build complete ==="
echo "Binary: ${BUILD_DIR}/blender_scraper"
echo ""
echo "Quick start:"
echo "  cp config.example.json config.json"
echo "  # Edit config.json and set your GitHub token"
echo "  ./${BUILD_DIR}/blender_scraper --help"
