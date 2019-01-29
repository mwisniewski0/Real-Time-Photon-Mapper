CMAKE_MINIMUM_REQUIRED(VERSION 3.3.13)

set(CMAKE_MACOSX_RPATH 1)

project(photon)

message("${CMAKE_SOURCE_DIR}")

add_subdirectory("${CMAKE_SOURCE_DIR}/external_libs/lib3ds")
include_directories("${CMAKE_SOURCE_DIR}/external_libs/lib3ds/src")
include_directories("${CMAKE_SOURCE_DIR}/external_libs/lib3ds/examples")
message("THESE ARE THE DIRECTORIES BEING INCLUDED")
#
#get_property(dirs DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
#foreach(dir ${dirs})
#    message(STATUS "dir='${dir}'")
#endforeach()

add_subdirectory("${CMAKE_SOURCE_DIR}/3ds_parser")
