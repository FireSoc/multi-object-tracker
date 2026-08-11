cmake_minimum_required(VERSION 3.24)

project(multi_object_tracker
    VERSION 0.1.0
    DESCRIPTION "Multi-object detection and tracking"
    LANGUAGES CXX)

# --- Compiler settings ------------------------------------------------------
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Generates build/compile_commands.json so clangd/Cursor understands the code.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Debug)
endif()

add_compile_options(-Wall -Wextra -Wpedantic)

# --- Dependencies -----------------------------------------------------------
# OpenCV: video capture, image processing, DNN inference (YOLO/COCO models).
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio highgui dnn video geometry)

# Eigen: linear algebra for the Kalman filter (matrices, vectors).
find_package(Eigen3 REQUIRED NO_MODULE)

message(STATUS "OpenCV version: ${OpenCV_VERSION}")

# --- Core library (src/ + include/) -----------------------------------------
# All real logic lives here: no main() in this target, ever.
# Add new files as you write them (kalman_filter.cpp, hungarian.cpp, ...).
add_library(tracker_core
    src/video_source.cpp
    src/motion_detector.cpp
)
target_include_directories(tracker_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(tracker_core PUBLIC ${OpenCV_LIBS} Eigen3::Eigen)

# --- Main executable (apps/) -------------------------------------------------
# Just a main() that wires up and runs the library.
add_executable(tracker apps/main.cpp)
target_link_libraries(tracker PRIVATE tracker_core)

# --- Examples / practice (examples/) -----------------------------------------
# Every .cpp in examples/ automatically becomes its own executable
# (each file must have its own main). No CMake edits needed for new files.
file(GLOB EXAMPLE_SOURCES CONFIGURE_DEPENDS examples/*.cpp)
foreach(example_src IN LISTS EXAMPLE_SOURCES)
    get_filename_component(example_name ${example_src} NAME_WE)
    add_executable(${example_name} ${example_src})
    target_include_directories(${example_name} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_link_libraries(${example_name} PRIVATE ${OpenCV_LIBS} Eigen3::Eigen)
endforeach()

# --- Tests ------------------------------------------------------------------
option(BUILD_TESTS "Build unit tests" ON)

if(BUILD_TESTS)
    include(FetchContent)
    FetchContent_Declare(
        googletest
        URL https://github.com/google/googletest/archive/refs/tags/v1.15.2.zip
    )
    FetchContent_MakeAvailable(googletest)

    enable_testing()

    # Add new test files here as you write them, e.g. tests/kalman_filter_test.cpp
    add_executable(tracker_tests
        tests/sanity_test.cpp
    )
    target_link_libraries(tracker_tests PRIVATE GTest::gtest_main tracker_core)

    include(GoogleTest)
    gtest_discover_tests(tracker_tests DISCOVERY_MODE PRE_TEST)
endif()
