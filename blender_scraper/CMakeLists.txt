cmake_minimum_required(VERSION 3.16)
project(blender_scraper CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- libcurl (required) ---
find_package(CURL REQUIRED)

# --- nlohmann/json ---
# Use vcpkg-installed version if available, otherwise fetch it
find_package(nlohmann_json CONFIG QUIET)
if(NOT nlohmann_json_FOUND)
    message(STATUS "nlohmann_json not found via vcpkg — fetching from GitHub")
    include(FetchContent)
    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG        v3.11.3
    )
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# --- Executables ---
add_executable(blender_scraper src/main.cpp)
add_executable(build_dataset   src/build_dataset.cpp)

foreach(TARGET blender_scraper build_dataset)
    target_include_directories(${TARGET} PRIVATE ${CMAKE_SOURCE_DIR}/src)
    target_link_libraries(${TARGET} PRIVATE CURL::libcurl nlohmann_json::nlohmann_json)
    if(MSVC)
        target_compile_options(${TARGET} PRIVATE /O2 /W3 /EHsc)
    else()
        target_compile_options(${TARGET} PRIVATE -O3 -Wall -Wextra -pthread
            $<$<CONFIG:Release>:-DNDEBUG>)
        target_link_libraries(${TARGET} PRIVATE pthread)
    endif()
endforeach()
