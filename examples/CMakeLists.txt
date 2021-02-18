cmake_minimum_required(VERSION 3.1.0)

set(PROJECT "SpidrExample")
PROJECT(${PROJECT})

find_package(OpenGL REQUIRED )
find_package(OpenMP REQUIRED)

get_filename_component(SPIDR_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
include_directories("${SPIDR_DIR}/include")
include_directories("${SPIDR_DIR}/external/HDILib/HDI/include")
include_directories("${SPIDR_DIR}/external/glfw/include/")

add_executable(${PROJECT} ArtificialImage.cpp)

# Set C++17 language standard
set_property(TARGET ${PROJECT} PROPERTY CXX_STANDARD 17)

# Prevent glfw including the OpenGL headers
# On Windows and Linux these come from glad in the HDILib
add_definitions(-DGLFW_INCLUDE_NONE)

# link to precompiled SpidrLib 
if(MSVC)
    MESSAGE( STATUS "Linking Windows libraries...")
    target_link_libraries(${PROJECT} debug "${SPIDR_DIR}/Build/Debug/Spidr.lib")
    target_link_libraries(${PROJECT} optimized "${SPIDR_DIR}/Build/Release/Spidr.lib")

    target_link_libraries(${PROJECT} debug "${SPIDR_DIR}/external/HDILib/HDI/Win/Debug/hdidimensionalityreduction.lib")
    target_link_libraries(${PROJECT} debug "${SPIDR_DIR}/external/HDILib/HDI/Win/Debug/hdidata.lib")
    target_link_libraries(${PROJECT} debug "${SPIDR_DIR}/external/HDILib/HDI/Win/Debug/hdiutils.lib")
    target_link_libraries(${PROJECT} optimized "${SPIDR_DIR}/external/HDILib/HDI/Win/Release/hdidimensionalityreduction.lib")
    target_link_libraries(${PROJECT} optimized "${SPIDR_DIR}/external/HDILib/HDI/Win/Release/hdidata.lib")
    target_link_libraries(${PROJECT} optimized "${SPIDR_DIR}/external/HDILib/HDI/Win/Release/hdiutils.lib")

    target_link_libraries(${PROJECT} debug "${SPIDR_DIR}/external/HDILib/Flann/Win/Debug/flann_cpp_s.lib")
    target_link_libraries(${PROJECT} optimized "${SPIDR_DIR}/external/HDILib/Flann/Win/Release/flann_cpp_s.lib")

    target_link_libraries(${PROJECT} debug "${SPIDR_DIR}/Build/external/glfw/src/Debug/glfw3.lib")
    target_link_libraries(${PROJECT} optimized "${SPIDR_DIR}//Build/external/glfw/src/Release/glfw3.lib")	

else()
	message(ERROR, "Only windows tested")    
endif(MSVC)

if(OPENMP_FOUND)
	message (STATUS "OpenMP found ${OpenMP_CXX_LIB_NAMES}")
	target_link_libraries(${PROJECT} OpenMP::OpenMP_CXX)
else()
    message(WARNING, "OpenMP not found!")    
endif()
