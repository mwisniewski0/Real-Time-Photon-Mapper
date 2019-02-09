cmake_minimum_required(VERSION 3.13)
project(photon)


set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

#builds the tinyobjloader library
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/external_libs/tinyobjloader")
#adds tinyobjloader to list of dirs that compiler uses to fine include files
include_directories("${CMAKE_CURRENT_SOURCE_DIR}/external_libs/tinyobjloader")

# builds the obj file parser
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/obj_parser")

message("THE COMPILER")
message("${CMAKE_CXX_COMPILER_ID}")