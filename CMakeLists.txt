cmake_minimum_required(VERSION 3.1.0)

set(PROJECT "Spidr")

PROJECT(${PROJECT})

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_INCLUDE_CURRENT_DIR ON)

if(MSVC)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /DWIN32 /EHsc /MP")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /NODEFAULTLIB:LIBCMT")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
endif(MSVC)

# use header-only libs, thus no components specified
find_package(Boost REQUIRED) 
find_package(Eigen3 3.3 REQUIRED)
find_package(OpenGL REQUIRED )
find_package(OpenMP REQUIRED)

set(LIB
	src/FeatureExtraction.h
	src/FeatureExtraction.cpp
	src/DistanceCalculation.h
	src/DistanceCalculation.cpp
	src/SpidrAnalysis.h
	src/SpidrAnalysis.cpp
	src/TsneComputation.h
	src/TsneComputation.cpp
	src/AnalysisParameters.h
)

set(UTILS
    src/KNNUtils.h
    src/KNNUtils.cpp
	src/FeatureUtils.h
	src/FeatureUtils.cpp
	src/EvalUtils.h
)

set(SOURCES ${LIB} ${UTILS})

source_group(Lib FILES ${LIB})
source_group(Utils FILES ${UTILS})

add_library(${PROJECT} SHARED ${SOURCES})

# Request C++17
set_property(TARGET ${PROJECT} PROPERTY CXX_STANDARD 17)

include_directories("external/hnswlib/")
include_directories("external/HDILib/HDI/include")
include_directories("external/spdlog/include/")
include_directories("external/glad/include/")
include_directories("external/glad/src/")
include_directories("external/glfw/include/")
include_directories(${Boost_INCLUDE_DIRS})
include_directories(${OPENGL_INCLUDE_DIRS} )

if(MSVC)
	ADD_DEFINITIONS(/arch:AVX)
	add_definitions(-DUSE_SSE)
	add_definitions(-DUSE_AVX)
    include_directories ("external/HDILib/Flann/Win/include")
endif(MSVC)

if(OPENMP_FOUND)
	message (STATUS "OpenMP found ${OpenMP_CXX_LIB_NAMES}")
	target_link_libraries(${PROJECT} OpenMP::OpenMP_CXX)
else()
    message(WARNING, "OpenMP not found!")    
endif()

target_link_libraries(${PROJECT} ${Boost_LIBRARIES})
target_link_libraries(${PROJECT} Eigen3::Eigen)

set( GLFW_BUILD_DOCS OFF CACHE BOOL  "GLFW lib only" )
set( GLFW_INSTALL OFF CACHE BOOL  "GLFW lib only" )
add_subdirectory("external/glfw")

if( MSVC )
    SET( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ENTRY:mainCRTStartup" )
endif()

# Prevent glfw including the OpenGL headers
# On Windows and Linux these come from glad in the HDILib
add_definitions(-DGLFW_INCLUDE_NONE)

target_link_libraries(${PROJECT} ${OPENGL_LIBRARIES} glfw )

#add_subdirectory("external/glad")
#target_link_libraries(${PROJECT} glad)

# link to precompiled HDILib 

if(MSVC)
    MESSAGE( STATUS "Linking Windows libraries...")
    target_link_libraries(${PROJECT} debug "${CMAKE_CURRENT_SOURCE_DIR}/external/HDILib/HDI/Win/Debug/hdidimensionalityreduction.lib")
    target_link_libraries(${PROJECT} debug "${CMAKE_CURRENT_SOURCE_DIR}/external/HDILib/HDI/Win/Debug/hdidata.lib")
    target_link_libraries(${PROJECT} debug "${CMAKE_CURRENT_SOURCE_DIR}/external/HDILib/HDI/Win/Debug/hdiutils.lib")
    target_link_libraries(${PROJECT} optimized "${CMAKE_CURRENT_SOURCE_DIR}/external/HDILib/HDI/Win/Release/hdidimensionalityreduction.lib")
    target_link_libraries(${PROJECT} optimized "${CMAKE_CURRENT_SOURCE_DIR}/external/HDILib/HDI/Win/Release/hdidata.lib")
    target_link_libraries(${PROJECT} optimized "${CMAKE_CURRENT_SOURCE_DIR}/external/HDILib/HDI/Win/Release/hdiutils.lib")

    target_link_libraries(${PROJECT} debug "${CMAKE_CURRENT_SOURCE_DIR}/external/HDILib/Flann/Win/Debug/flann_cpp_s.lib")
    target_link_libraries(${PROJECT} optimized "${CMAKE_CURRENT_SOURCE_DIR}/external/HDILib/Flann/Win/Release/flann_cpp_s.lib")
else()
	message(ERROR, "Only windows tested")    
endif(MSVC)

