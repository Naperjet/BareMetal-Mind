cmake_minimum_required(VERSION 3.10)
project(BareMetalNeural VERSION 0.1 LANGUAGES CXX)

# ---------------------------------------------------------------------------
# Language settings
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Default to Release when no build type is given (single-config generators)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# ---------------------------------------------------------------------------
# Library: perceptron + dataset generators
# ---------------------------------------------------------------------------
add_library(perceptron STATIC
    src/perceptron.cpp
    src/dataset.cpp
)
target_include_directories(perceptron PUBLIC src)

if(MSVC)
    target_compile_options(perceptron PRIVATE /W4)
else()
    target_compile_options(perceptron PRIVATE -Wall -Wextra -Wpedantic)
endif()

# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------
add_executable(perceptron_demo examples/perceptron_demo.cpp)
target_link_libraries(perceptron_demo PRIVATE perceptron)

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
add_executable(perceptron_tests tests/perceptron_tests.cpp)
target_link_libraries(perceptron_tests PRIVATE perceptron)

# The tests rely on assert(); keep it active even in Release (-DNDEBUG) builds.
if(MSVC)
    target_compile_options(perceptron_tests PRIVATE /UNDEBUG)
else()
    target_compile_options(perceptron_tests PRIVATE -UNDEBUG)
endif()

enable_testing()
add_test(NAME perceptron_tests COMMAND perceptron_tests)
