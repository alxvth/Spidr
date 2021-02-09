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
find_package(OpenMP REQUIRED)

set(LIB
	src/FeatureExtraction.h
	src/FeatureExtraction.cpp
	src/DistanceCalculation.h
	src/DistanceCalculation.cpp
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

include_directories ("external/hnswlib/")
include_directories ("external/HDILib/")
include_directories ("external/spdlog/include/")
include_directories(${Boost_INCLUDE_DIRS})

if(MSVC)
	ADD_DEFINITIONS(/arch:AVX)
	add_definitions(-DUSE_SSE)
	add_definitions(-DUSE_AVX)
endif(MSVC)

# Request C++17, in order to use std::for_each_n with std::execution::par_unseq.
set_property(TARGET ${PROJECT} PROPERTY CXX_STANDARD 17)

find_package(OpenMP REQUIRED)

if(OPENMP_FOUND)
	message (STATUS "OpenMP found ${OpenMP_CXX_LIB_NAMES}")
	target_link_libraries(${PROJECT} OpenMP::OpenMP_CXX)
else()
    message(WARNING, "OpenMP not found!")    
endif()

target_link_libraries(${PROJECT} ${Boost_LIBRARIES})
target_link_libraries(${PROJECT} Eigen3::Eigen)
