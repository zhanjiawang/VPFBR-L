cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(face_registion)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-std=c++11")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

list( APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules )
find_package(Eigen3 3.3.4 QUIET REQUIRED NO_MODULE)
find_package(Boost COMPONENTS program_options REQUIRED )
find_package(PCL 1.7)
find_package(Ceres REQUIRED )

include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
 
include_directories( ${CERES_INCLUDE_DIRS} )

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable (VPFBR VPFBR.cpp)
target_link_libraries (VPFBR ${CERES_LIBRARIES} ${PCL_LIBRARIES})

add_executable (VPFBL VPFBL.cpp)
target_link_libraries (VPFBL ${CERES_LIBRARIES} ${PCL_LIBRARIES})
