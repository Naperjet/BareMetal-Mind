cmake_minimum_required(VERSION 3.10)
project(BareMetalNeural CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
    target_compile_options(perceptron PRIVATE /W4)
else()
    target_compile_options(perceptron PRIVATE -Wall -Wextra -Wpedantic)
endif()

add_executable(perceptron_demo examples/perceptron_demo.cpp)
target_link_libraries(perceptron_demo PRIVATE perceptron)

add_executable(perceptron_tests tests.perceptron_tests.cpp)
target_link_libraries(perceptron_tests PRIVATE perceptron)

enable_testing()
if(NOT MSVC)
    target_compile_options(perceptron_tests PRIVATE -UNDEBUG)
endif()

add_test(NAME perceptron_tests COMMAND perceptron_tests)
