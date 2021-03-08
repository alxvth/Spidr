cmake_minimum_required(VERSION 3.17)

set(SPIDRLIB "SpidrLib")

PROJECT(${SPIDRLIB})

# use header-only libs, thus no components specified
find_package(Boost REQUIRED) 
find_package(Eigen3 3.3 REQUIRED)
find_package(OpenGL REQUIRED )
find_package(OpenMP REQUIRED)

set(HEADERS 
	include/FeatureExtraction.h
	include/DistanceCalculation.h
	include/SpidrAnalysis.h
	include/TsneComputation.h
	include/SpidrAnalysisParameters.h
	include/KNNUtils.h
	include/FeatureUtils.h
	include/EvalUtils.h
	include/KNNDists.h
)

set(LIB
	src/FeatureExtraction.cpp
	src/DistanceCalculation.cpp
	src/SpidrAnalysis.cpp
	src/TsneComputation.cpp
)

set(UTILS
	src/KNNUtils.cpp
	src/FeatureUtils.cpp
	src/EvalUtils.cpp
)

set(SOURCES ${HEADERS} ${LIB} ${UTILS})

source_group(Headers FILES ${HEADERS})
source_group(Lib FILES ${LIB})
source_group(Utils FILES ${UTILS})

add_library(${SPIDRLIB} STATIC ${SOURCES})

# Request C++17
target_compile_features(${SPIDRLIB} PRIVATE cxx_std_17)

target_include_directories(${SPIDRLIB} PUBLIC "include")
target_include_directories(${SPIDRLIB} PUBLIC "external/HDILib/HDI/include")
target_include_directories(${SPIDRLIB} PUBLIC ${Boost_INCLUDE_DIRS})

if(MSVC)
	ADD_DEFINITIONS(/arch:AVX)
	add_definitions(-DUSE_SSE)
	add_definitions(-DUSE_AVX)
    target_include_directories(${SPIDRLIB} PUBLIC "external/HDILib/Flann/Win/include")
endif(MSVC)

if(OPENMP_FOUND)
	message (STATUS "OpenMP found ${OpenMP_CXX_LIB_NAMES}")
	target_link_libraries(${SPIDRLIB} PUBLIC OpenMP::OpenMP_CXX)
else()
    message(WARNING, "OpenMP not found!")    
endif()

set(GLFW_BUILD_DOCS OFF CACHE BOOL "GLFW lib only" )
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "disable GLFX examples")
set(GLFW_BUILD_TESTS OFF CACHE BOOL "disable GLFX tests") 
set(GLFW_INSTALL OFF CACHE BOOL "GLFW lib only" )
add_subdirectory("external/glfw")

if( MSVC )
    SET( CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} /ENTRY:mainCRTStartup" )
endif()

# Prevent glfw including the OpenGL headers
# On Windows and Linux these come from glad in the HDILib
add_definitions(-DGLFW_INCLUDE_NONE)

target_include_directories(${SPIDRLIB} PUBLIC "${CMAKE_SOURCE_DIR}/external/glfw/include")
target_link_libraries(${SPIDRLIB} PUBLIC glfw)

target_include_directories(${SPIDRLIB} PUBLIC "external/spdlog/include")
target_include_directories(${SPIDRLIB} PUBLIC "external/hnswlib")

target_link_libraries(${SPIDRLIB} PUBLIC ${Boost_LIBRARIES})
target_link_libraries(${SPIDRLIB} PUBLIC Eigen3::Eigen)

# link to precompiled HDILib 
if(MSVC)
    MESSAGE( STATUS "Linking Windows libraries...")
	
    target_link_libraries(${SPIDRLIB} PUBLIC debug "${CMAKE_CURRENT_SOURCE_DIR}/external/HDILib/HDI/Win/Debug/hdidimensionalityreduction.lib")
    target_link_libraries(${SPIDRLIB} PUBLIC debug "${CMAKE_CURRENT_SOURCE_DIR}/external/HDILib/HDI/Win/Debug/hdidata.lib")
    target_link_libraries(${SPIDRLIB} PUBLIC debug "${CMAKE_CURRENT_SOURCE_DIR}/external/HDILib/HDI/Win/Debug/hdiutils.lib")
    target_link_libraries(${SPIDRLIB} PUBLIC optimized "${CMAKE_CURRENT_SOURCE_DIR}/external/HDILib/HDI/Win/Release/hdidimensionalityreduction.lib")
    target_link_libraries(${SPIDRLIB} PUBLIC optimized "${CMAKE_CURRENT_SOURCE_DIR}/external/HDILib/HDI/Win/Release/hdidata.lib")
    target_link_libraries(${SPIDRLIB} PUBLIC optimized "${CMAKE_CURRENT_SOURCE_DIR}/external/HDILib/HDI/Win/Release/hdiutils.lib")

    target_link_libraries(${SPIDRLIB} PUBLIC debug "${CMAKE_CURRENT_SOURCE_DIR}/external/HDILib/Flann/Win/Debug/flann_cpp_s.lib")
    target_link_libraries(${SPIDRLIB} PUBLIC optimized "${CMAKE_CURRENT_SOURCE_DIR}/external/HDILib/Flann/Win/Release/flann_cpp_s.lib")
	
else()
	message(ERROR, "Only windows tested")    
endif(MSVC)

