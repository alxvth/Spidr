cmake_minimum_required(VERSION 3.1.0)

set(PROJECT "SpidrExample")
PROJECT(${PROJECT})

add_executable(${PROJECT} SpidrExample.cpp)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../ ${CMAKE_CURRENT_BINARY_DIR}/SpidrLib)
target_link_libraries(${PROJECT} PUBLIC SpidrLib)

# On Windows and Linux these come from glad in the HDILib which is linked against in the SpidrLib
# Prevent glfw including the OpenGL headers. It's important to define this AFTER linking against SpidrLib
add_definitions(-DGLFW_INCLUDE_NONE)

# Set C++17 language standard
target_compile_features(${PROJECT} PUBLIC cxx_std_17)
