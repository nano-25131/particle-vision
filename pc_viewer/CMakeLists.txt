cmake_minimum_required(VERSION 3.18)
project(cpp_viewer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE RelWithDebInfo)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -g")

# ---- dependencies ----
find_package(OpenCV REQUIRED)
find_package(glfw3 REQUIRED)

# ---- cpp-httplib + nlohmann/json (header-only) ----
set(EXTLIB_DIR ${CMAKE_SOURCE_DIR}/../particle_system/../tensorrt-unet3plus-jetson-main/trt_run)

add_executable(viewer
    main.cpp
    implot/implot.cpp
    implot/implot_items.cpp
)

target_include_directories(viewer PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${EXTLIB_DIR}
    ${CMAKE_SOURCE_DIR}/implot
    /usr/include/imgui
    /usr/include/imgui/backends
)

target_link_libraries(viewer
    ${OpenCV_LIBS}
    glfw
    GL
    imgui
    stb
    pthread
    ssl
    crypto
)
