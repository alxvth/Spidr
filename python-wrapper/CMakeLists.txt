cmake_minimum_required(VERSION 3.17)

set(SPIDRWRAPPER "SpidrWrapper")
PROJECT(${SPIDRWRAPPER})

# add python wrapper lib
add_subdirectory(pybind11)
pybind11_add_module(${SPIDRWRAPPER} SpidrWrapper.cpp)

# add Spidr lib
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../ ${CMAKE_CURRENT_BINARY_DIR}/SpidrLib)
target_link_libraries(${SPIDRWRAPPER} PUBLIC SpidrLib)

# On Windows and Linux these come from glad in the HDILib which is linked against in the SpidrLib
# Prevent glfw including the OpenGL headers. It's important to define this AFTER linking against SpidrLib
add_definitions(-DGLFW_INCLUDE_NONE)

# Set C++17 language standard
target_compile_features(${SPIDRWRAPPER} PUBLIC cxx_std_17)

# EXAMPLE_VERSION_INFO is defined by setup.py and passed into the C++ code as a
# define (VERSION_INFO) here.
target_compile_definitions(${SPIDRWRAPPER} PRIVATE VERSION_INFO=${EXAMPLE_VERSION_INFO})